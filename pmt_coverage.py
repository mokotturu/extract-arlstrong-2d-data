import os
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

def getIDs(fileName: str) -> np.ndarray:
	'''
	extracts the UUIDs from the exported MTurk batch file

	## Parameters

	fileName: str

	Name of the batch file to be read
	'''
	df = pd.read_csv(fileName)
	surveyCodes = df['Answer.surveycode']
	return surveyCodes.to_numpy()

def getDataHelper(ids: np.ndarray, db) -> pd.DataFrame:
	'''
	Returns a Pandas DataFrame object with the exploration data for given participant UUIDs

	## Parameters

	ids: np.ndarray

	Array of the UUIDs of the participants

	db: pymongo.database.Database

	PyMongo Database object from which data should be queried
	'''
	# constant
	totalCells = 54801

	# get explored cells data for each participant
	data = list(db.simulationresults.aggregate([
		{
			'$match': {
				'uuid': {
					'$in': list(ids)
				}
			},
		}, {
			'$project': {
				'uuid': 1,
				'section2.humanExplored': 1,
			}
		}
	]))

	uuidArr = []
	numAbsCellsArr = np.zeros(len(data))
	percentArr = np.zeros(len(data))
	totalCellsArr = np.full(len(data), totalCells)

	# store exploration metrics for each participants
	for i, participantData in enumerate(data):
		uuidArr.append(participantData['uuid'])
		numAbsCellsArr[i] = len(participantData['section2']['humanExplored'])
		percentArr[i] = numAbsCellsArr[i] / totalCells

	return pd.DataFrame(
		data = {
			'uuid': uuidArr,
			'Number of grid cells explored': numAbsCellsArr,
			'Percentage of explorable area covered': percentArr,
			'Total number of grid cells': totalCellsArr,
		}
	)

def main():
	print('Do you want to combine all the batch files? (Y/n) ', end='')
	shouldCombine = input().capitalize() != 'N'

	# connect to DB
	load_dotenv(verbose=True)
	MONGO_URI = os.environ['MONGO_URI']
	client = MongoClient(MONGO_URI)
	db = client.tempDB

	if shouldCombine:
		print("Combining all batch files...")
		ids = np.array([])
		for file in os.listdir('batches/input'):
			ids = np.append(ids, getIDs(f'batches/input/{file}'))

		res = getDataHelper(ids, db)
		res.to_csv(f'batches/output/Data_combined_{datetime.now().strftime("%y%m%d%H%M%S")}.csv')
	else:
		print("Keeping batch files separate...")
		for file in os.listdir('batches/input'):
			ids = getIDs(f'batches/input/{file}')
			res = getDataHelper(ids, db)
			res.to_csv(f'batches/output/Data_for_{file}')

if __name__ == '__main__':
	main()
