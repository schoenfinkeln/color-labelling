import os, sys, json
import pandas as pd
from pathlib import Path

def main(argv):
	if len(argv) == 0:
		dir = "./"
	else:
		dir = argv[0]
	if not argv[0].endswith('/'):
		dir += '/'
	file_list = [Path(f).stem for f in os.listdir(dir) if f.endswith('.json')]
	titles = pd.read_csv(argv[1])

	cv_list = []
	for item in file_list:
		tmp = {}
		tmp['id'] = item
		item = item + '.jpg'
		tmp['title'] = titles[titles['path'] == item].title.values[0]
		cv_list.append(tmp)

	with open('./file_list.json', 'w') as outfile:
		json.dump(cv_list, outfile)

if __name__ == "__main__":
	main(sys.argv[1:])