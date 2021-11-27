import re
import os
import glob

# this is a negative lookbehind expression that finds all
# path references with images/
IMAGE_REGEX = re.compile(r'\.{0,2}\/?(?:images\/)(\w+\.(?:png|svg))')
GITHUB_IMAGE_PREFIX = "https://raw.githubusercontent.com/ychennay/dso-560-nlp-text-analytics/main/images/"

def replace_line(line: str)-> str:
	match = re.search(IMAGE_REGEX, line)
	if match:
		image_name = match.group(1)
		full_path = match.group()
		
		new_path = GITHUB_IMAGE_PREFIX + image_name
		return line.replace(full_path, new_path)
	return line

def replace_image_paths()-> None:
	for week_number in range(1, 6):
		os.chdir(f"../week{week_number}")
		for jupyter_notebook_file in glob.glob("*.ipynb"):
			with open(jupyter_notebook_file, mode="r") as file_desc:
				lines = file_desc.readlines()
				new_lines = list(map(replace_line, lines))
			with open(f"{jupyter_notebook_file}", mode="w") as new_file_desc:
				new_file_desc.writelines(new_lines)
			input(f"Done with {jupyter_notebook_file}")

if __name__ == "__main__":
	replace_image_paths()