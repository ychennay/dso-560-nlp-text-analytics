import pandas as pd
from pyspark.mllib.recommendation import ALS, Rating
from pyspark.sql import SparkSession, SQLContext
from sklearn.metrics.pairwise import cosine_similarity
if __name__ == "__main__":  # run this by typing "python collaborative_filter.py"
    app_name = "collab_filter_example"

    # create a Spark context
    spark = SparkSession.builder.master("local").appName(app_name).getOrCreate()

    # create a Spark SQL context to allow us to run SQL commands
    sql_context = SQLContext(spark.sparkContext)

    df = spark.read.csv("ratings_small.csv", header=True, sep=",")
    print(df)  # DataFrame[userId: string, movieId: string, rating: string, timestamp: string]
    df.createOrReplaceTempView("ratings")
    print(df.count())

    # we'll filter out ratings from users with less than 10 ratings, and from films with less than 20 ratings
    df = sql_context.sql("SELECT * "
                         "FROM ratings "
                         "WHERE userID IN (SELECT userID "
                         "FROM ratings GROUP BY userID HAVING COUNT(*) >= 10) "
                         "AND movieID IN (SELECT movieID FROM ratings GROUP BY movieID "
                         "HAVING COUNT(*) >= 20)")

    # load in the movies names
    movies = spark.read.csv("movies.csv", header=True, sep=",")
    print(movies)  # DataFrame[userId: string, movieId: string, rating: string]
    movies.createOrReplaceTempView("movies")  # create a SQL table called movies

    # the Alternating Least Squares model that PySpark uses requires you to have a Rating object for each
    # row, with the user ID, product ID, and rating as the three columns. So I am mapping the dataframe from a
    # regular dataframe into an RDD (resilient distributed dataset) of Ratings objects
    ratings = df.rdd.map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
    print(ratings)  # PythonRDD[13] at RDD at PythonRDD.scala:53

    # at this point, conceptually, we have a matrix that is U x P. U is the # of distinct users. P is the # of distinct
    # products, or films, in this case.

    rank = 10  # this is the number of dimensions D we want to reduce down to
    numIterations = 15

    # In collaborative filter, there is a U x P original matrix that is made up of two smaller
    # U x D and D x P matrices. The D represents the number of reduced dimensions - in this case 20 (the rank variable).
    # Also remember that the original U x P matrix has lots of missing values in it, since most users have not
    # watched/rated most films. The ALS model will iteratively try to update the values in the U x D and D x P matrices
    # until they match as close to possible the known values in the U x P (the original user-product ratings matrix).

    model = ALS.train(ratings, rank, numIterations)

    # Now that the model has finished, we have two new completely updated matrices: U x D and D x P. We care about the
    # D x P matrix. This basically represents out reduced dimensions for each product (film). For each film, we get this
    # vector. It will be of size 20.

    # get the film features (each row will be a tuple - (film_id, array of features representing film size 20)
    film_features = model.productFeatures()
    spark.createDataFrame(film_features) \
        .toDF("film_id", "features") \
        .createOrReplaceTempView("film_features")  # from the film_features, create a sql table called product_features

    pandas_film_features_df = sql_context.sql("SELECT m.original_title as film, ff.features "
                                              "FROM film_features ff "
                                              "JOIN movies m ON m.id = ff.film_id").toPandas()

    print(pandas_film_features_df)  # now it's just another normal pandas dataframe, with a film column
    # that contains the filmn name, a column with the film ID, and another column with numpy arrays representing the
    # reduced dimensional vector that represents a film

    film_names = list(pandas_film_features_df["film"].values)

    film_similarities = pd.DataFrame(cosine_similarity(list(pandas_film_features_df["features"].values)), index=film_names,
                                     columns=film_names).transpose()

    similarities_df = film_similarities.unstack().reset_index()

    similarities_df.columns = ["film1", "film2", "similarity"]
    similarities_df = similarities_df[similarities_df["similarity"] < 0.99999999]
    similarities_df = similarities_df[similarities_df["similarity"] >= 0.50]
    similarities_df.sort_values(by="similarity", ascending=False, inplace=True)
    similarities_df.to_csv("similarities.csv")
