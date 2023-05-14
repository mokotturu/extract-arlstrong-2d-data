import os
from collections import OrderedDict
from datetime import datetime
from pprint import pprint

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient


def getIDs(fileName: str) -> np.ndarray:
	'''
	extracts the UUIDs from the exported MTurk batch file

	## Parameters

	fileName: Name of the batch file to be read
	'''
	df = pd.read_csv(fileName)
	surveyCodes = df['Answer.surveycode']
	return surveyCodes.to_numpy()

def getDataHelper(url: str, db) -> pd.DataFrame:
	'''
	Returns a Pandas DataFrame object with the game data for given participant UUIDs

	## Parameters

	ids: Array of the UUIDs of the participants

	db: PyMongo Database object from which data should be queried
	'''
	data = list(db.simulationresults.aggregate([
		{
			'$match': {
				'playedOn': url,
				'createdAt': {
					'$gte': datetime(2023, 4, 25, 0, 0, 0, 0),
					'$lt': datetime(2023, 4, 28, 0, 0, 0, 0),
				}
			},
		}, {
			'$project': {
				'uuid': 1,
				'section2.humanExplored': 1,
				'decisions.agent1': 1,
				'failedTutorial': 1,
				'createdAt': 1,
				'survey1Modified': 1,
				'survey2Modified': 1,
				'gameMode': 1,
				'endGame': 1,
				'survey1': 1,
				'survey2': 1,
			}
		}
	]))

	print(f'Number of participants: {len(data)}')

	# constants
	TOTALCELLS = 54801
	INTERVALS = 7

	dataObj = OrderedDict()

	# identification and general game stats
	dataObj['uuid'] = []
	dataObj['createdAt'] = []
	dataObj['gameMode'] = []
	dataObj['failedTutorial'] = []
	dataObj['survey1Modified'] = []
	dataObj['survey2Modified'] = []

	# coverage stats
	dataObj['Number of grid cells explored'] = []
	dataObj['Percentage of explorable area covered'] = []
	dataObj['Total number of grid cells'] = [TOTALCELLS for _ in range(len(data))]

	# decisions and other per-round stats
	for i in range(INTERVALS):
		dataObj[f'decisions{i}'] = []
		dataObj[f'timeTaken{i}'] = []
		dataObj[f'humanGoldTargetsCollected{i}'] = []
		dataObj[f'performanceRating{i}'] = []
		dataObj[f'honestlyMoralityRating{i}'] = []
		dataObj[f'influenceText{i}'] = []

	# end game survey stats
	dataObj['endGameRoundsPlayed'] = []
	dataObj['endGameLastTwoRounds'] = []

	# post game survey stats
	dataObj['survey1Reliable'] = []
	dataObj['survey1Compentent'] = []
	dataObj['survey1Ethical'] = []
	dataObj['survey1Transparent'] = []
	dataObj['survey1Benevolent'] = []
	dataObj['survey1Predictable'] = []
	dataObj['survey1Skilled'] = []
	dataObj['survey1Principled'] = []
	dataObj['survey1Genuine'] = []
	dataObj['survey1Kind'] = []
	dataObj['survey1SelectThree'] = []
	dataObj['survey1Dependable'] = []
	dataObj['survey1Capable'] = []
	dataObj['survey1Moral'] = []
	dataObj['survey1Sincere'] = []
	dataObj['survey1Considerate'] = []
	dataObj['survey1Consistent'] = []
	dataObj['survey1Meticulous'] = []
	dataObj['survey1HasIntegrity'] = []
	dataObj['survey1Candid'] = []
	dataObj['survey1Goodwill'] = []
	dataObj['survey2Gender'] = []
	dataObj['survey2SelfDescribeText'] = []
	dataObj['survey2Age'] = []
	dataObj['survey2Education'] = []
	dataObj['survey2TechEd'] = []
	dataObj['survey2RoboticsExp'] = []

	for i, participantData in enumerate(data):
		try:
			dataObj['uuid'].append(participantData['uuid'])
		except:
			dataObj['uuid'].append('')

		try:
			dataObj['createdAt'].append(str(participantData['createdAt']))
		except:
			dataObj['createdAt'].append('')

		try:
			dataObj['gameMode'].append(participantData['gameMode'])
		except:
			dataObj['gameMode'].append('')

		try:
			dataObj['failedTutorial'].append(participantData['failedTutorial'])
		except:
			dataObj['failedTutorial'].append('')

		try:
			dataObj['survey1Modified'].append(participantData['survey1Modified'])
		except:
			dataObj['survey1Modified'].append('')

		try:
			dataObj['survey2Modified'].append(participantData['survey2Modified'])
		except:
			dataObj['survey2Modified'].append('')

		dataObj['Number of grid cells explored'].append(len(participantData['section2']['humanExplored']))
		dataObj['Percentage of explorable area covered'].append(dataObj['Number of grid cells explored'][i] / TOTALCELLS)

		if participantData['decisions']['agent1']:
			for j, stepObj in enumerate(participantData['decisions']['agent1']):
				surveyRes = {'performanceRating': '', 'honestlyMoralityRating': '', 'influenceText': ''}
				for obj in stepObj['surveyResponse']:
					if obj['name'] == 'performanceRating':
						surveyRes['performanceRating'] = obj['value']
					elif obj['name'] == 'honestlyMoralityRating':
						surveyRes['honestlyMoralityRating'] = obj['value']
					elif obj['name'] == 'influenceText':
						surveyRes['influenceText'] = obj['value']

				dataObj[f'decisions{j}'].append(stepObj['decision'])
				dataObj[f'timeTaken{j}'].append(stepObj['timeTaken'])
				dataObj[f'humanGoldTargetsCollected{j}'].append(stepObj['humanGoldTargetsCollected'])
				dataObj[f'performanceRating{j}'].append(surveyRes['performanceRating'])
				dataObj[f'honestlyMoralityRating{j}'].append(surveyRes['honestlyMoralityRating'])
				dataObj[f'influenceText{j}'].append(surveyRes['influenceText'])
		else:
			for j in range(INTERVALS):
				dataObj[f'decisions{j}'].append('')
				dataObj[f'timeTaken{j}'].append('')
				dataObj[f'humanGoldTargetsCollected{j}'].append('')
				dataObj[f'performanceRating{j}'].append('')
				dataObj[f'honestlyMoralityRating{j}'].append('')
				dataObj[f'influenceText{j}'].append('')

		if len(participantData['endGame']) > 0:
			dataObj['endGameRoundsPlayed'].append(participantData['endGame'][0]['value'])
			dataObj['endGameLastTwoRounds'].append(participantData['endGame'][1]['value'])
		else:
			dataObj['endGameRoundsPlayed'].append('')
			dataObj['endGameLastTwoRounds'].append('')

		try:
			if participantData['survey1']:
				dataObj['survey1Reliable'].append(participantData['survey1']['reliable'])
				dataObj['survey1Compentent'].append(participantData['survey1']['competent'])
				dataObj['survey1Ethical'].append(participantData['survey1']['ethical'])
				dataObj['survey1Transparent'].append(participantData['survey1']['transparent'])
				dataObj['survey1Benevolent'].append(participantData['survey1']['benevolent'])
				dataObj['survey1Predictable'].append(participantData['survey1']['predictable'])
				dataObj['survey1Skilled'].append(participantData['survey1']['skilled'])
				dataObj['survey1Principled'].append(participantData['survey1']['principled'])
				dataObj['survey1Genuine'].append(participantData['survey1']['genuine'])
				dataObj['survey1Kind'].append(participantData['survey1']['kind'])
				dataObj['survey1SelectThree'].append(participantData['survey1']['selectThree'])
				dataObj['survey1Dependable'].append(participantData['survey1']['dependable'])
				dataObj['survey1Capable'].append(participantData['survey1']['capable'])
				dataObj['survey1Moral'].append(participantData['survey1']['moral'])
				dataObj['survey1Sincere'].append(participantData['survey1']['sincere'])
				dataObj['survey1Considerate'].append(participantData['survey1']['considerate'])
				dataObj['survey1Consistent'].append(participantData['survey1']['consistent'])
				dataObj['survey1Meticulous'].append(participantData['survey1']['meticulous'])
				dataObj['survey1HasIntegrity'].append(participantData['survey1']['hasintegrity'])
				dataObj['survey1Candid'].append(participantData['survey1']['candid'])
				dataObj['survey1Goodwill'].append(participantData['survey1']['goodwill'])
		except:
			dataObj['survey1Reliable'].append('')
			dataObj['survey1Compentent'].append('')
			dataObj['survey1Ethical'].append('')
			dataObj['survey1Transparent'].append('')
			dataObj['survey1Benevolent'].append('')
			dataObj['survey1Predictable'].append('')
			dataObj['survey1Skilled'].append('')
			dataObj['survey1Principled'].append('')
			dataObj['survey1Genuine'].append('')
			dataObj['survey1Kind'].append('')
			dataObj['survey1SelectThree'].append('')
			dataObj['survey1Dependable'].append('')
			dataObj['survey1Capable'].append('')
			dataObj['survey1Moral'].append('')
			dataObj['survey1Sincere'].append('')
			dataObj['survey1Considerate'].append('')
			dataObj['survey1Consistent'].append('')
			dataObj['survey1Meticulous'].append('')
			dataObj['survey1HasIntegrity'].append('')
			dataObj['survey1Candid'].append('')
			dataObj['survey1Goodwill'].append('')

		try:
			if participantData['survey2']:
				dataObj['survey2Gender'].append(participantData['survey2']['gender'])
				dataObj['survey2SelfDescribeText'].append(participantData['survey2']['selfDescribeText'])
				dataObj['survey2Age'].append(participantData['survey2']['age'])
				dataObj['survey2Education'].append(participantData['survey2']['education'])
				dataObj['survey2TechEd'].append(participantData['survey2']['techEd'])
				dataObj['survey2RoboticsExp'].append(participantData['survey2']['roboticsExp'])
		except:
			dataObj['survey2Gender'].append('')
			dataObj['survey2SelfDescribeText'].append('')
			dataObj['survey2Age'].append('')
			dataObj['survey2Education'].append('')
			dataObj['survey2TechEd'].append('')
			dataObj['survey2RoboticsExp'].append('')

	return pd.DataFrame(data=dataObj)

def getProlificIDs():
	pass

def main():
	# connect to DB
	load_dotenv(verbose=True)
	MONGO_URI = os.environ['MONGO_URI']
	client = MongoClient(MONGO_URI)
	db = client.tempDB

	url = "arlstrong-uml-034-prolific.herokuapp.com"
	res = getDataHelper(url, db)
	res.to_csv(f'batches/output/Data_for_PTV_2.csv')

	print('Completed generating csv files.')

if __name__ == '__main__':
	main()
