#include "Settings.h"

#if EXAMPLE_SELECTION == EXAMPLE_KAGGLE

#include <neo/PredictiveHierarchy.h>

#include <time.h>
#include <iostream>
#include <random>
#include <fstream>
#include <cmath>
#include <string>
#include <sstream>

#include <unordered_map>
#include <unordered_set>

#include <algorithm>
#include <memory>

#include <assert.h>

class Entry {
public:
	int _id;
	int _store;
	int _dayOfWeek;
	int _year, _month, _day;
	int _sales;
	int _customers;
	bool _open;
	bool _promo;
	bool _stateHoliday_public, _stateHoliday_easter, _stateHoliday_christmas;
	bool _schoolHoliday;

	bool operator<(const Entry &other) {
		if (_year < other._year)
			return true;
		else if (_year > other._year)
			return false;
		else {
			if (_month < other._month)
				return true;
			else if (_month > other._month)
				return false;
			else {
				if (_day < other._day)
					return true;
				else if (_day > other._day)
					return false;
				else
					return false;
			}
		}

		return false;
	}
};

bool compareEntryDates(const std::unique_ptr<Entry> &lhs, const std::unique_ptr<Entry> &rhs) {
	return (*lhs) < (*rhs);
}

void loadData(const std::string &fileNameTrain, const std::string &fileNameTest, std::vector<std::unique_ptr<Entry>> &entries) {
	// Load training data (id is -1)
	{
		std::ifstream fromTrain(fileNameTrain);

		std::string line = "";

		// Skip first line
		std::getline(fromTrain, line);

		do {
			Entry e;

			e._id = -1;

			std::getline(fromTrain, line);

			if (line == "")
				break;

			// Get data
			std::vector<std::string> data;

			std::istringstream fromLine(line);

			std::vector<std::string> strVals;

			do {
				std::string strVal;
				std::getline(fromLine, strVal, ',');

				if (strVal == "")
					break;

				strVals.push_back(strVal);

			} while (fromLine.good() && !fromLine.eof());

			assert(strVals.size() == 9);

			// Convert
			e._store = std::stoi(strVals[0]);
			e._dayOfWeek = std::stoi(strVals[1]);

			std::istringstream fromDate(strVals[2]);

			std::string datePart;

			std::getline(fromDate, datePart, '-');
			e._year = std::stoi(datePart);

			std::getline(fromDate, datePart, '-');
			e._month = std::stoi(datePart);

			std::getline(fromDate, datePart, '-');
			e._day = std::stoi(datePart);

			e._sales = std::stoi(strVals[3]);
			e._customers = std::stoi(strVals[4]);
			e._open = std::stoi(strVals[5]);
			e._promo = std::stoi(strVals[6]);

			std::string holiday = strVals[7].substr(1, 1);

			e._stateHoliday_public = holiday == "a";
			e._stateHoliday_easter = holiday == "b";
			e._stateHoliday_christmas = holiday == "c";

			e._schoolHoliday = std::stoi(strVals[8].substr(1, 1));

			entries.push_back(std::make_unique<Entry>(e));
		} while (fromTrain.good() && !fromTrain.eof());
	}

	// Load test data
	{
		std::ifstream fromTest(fileNameTest);

		std::string line = "";

		// Skip first line
		std::getline(fromTest, line);

		do {
			Entry e;

			std::getline(fromTest, line);

			if (line == "")
				break;

			// Get data
			std::vector<std::string> data;

			std::istringstream fromLine(line);

			std::vector<std::string> strVals;

			do {
				std::string strVal;
				std::getline(fromLine, strVal, ',');

				if (strVal == "")
					break;

				strVals.push_back(strVal);

			} while (fromLine.good() && !fromLine.eof());

			assert(strVals.size() == 9);

			// Convert
			e._id = std::stoi(strVals[0]);
			e._store = std::stoi(strVals[1]);
			e._dayOfWeek = std::stoi(strVals[2]);

			std::istringstream fromDate(strVals[3]);

			std::string datePart;

			std::getline(fromDate, datePart, '-');
			e._year = std::stoi(datePart);

			std::getline(fromDate, datePart, '-');
			e._month = std::stoi(datePart);

			std::getline(fromDate, datePart, '-');
			e._day = std::stoi(datePart);

			e._sales = std::stoi(strVals[4]);
			e._customers = std::stoi(strVals[5]);
			e._open = std::stoi(strVals[6]);
			e._promo = std::stoi(strVals[7]);

			std::string holiday = strVals[8].substr(1, 1);

			e._stateHoliday_public = holiday == "a";
			e._stateHoliday_easter = holiday == "b";
			e._stateHoliday_christmas = holiday == "c";

			e._schoolHoliday = std::stoi(strVals[9].substr(1, 1));

			entries.push_back(std::make_unique<Entry>(e));
		} while (fromTest.good() && !fromTest.eof());
	}
}

int main() {
	std::mt19937 generator(time(nullptr));

	std::vector<std::unique_ptr<Entry>> entries;

	loadData("train/train.csv", "test/test.csv", entries);

	// Sort into time series (interleaved with test set)
	std::stable_sort(entries.begin(), entries.end());

	// Meta data
	std::unordered_set<int> stores;
	
	int maxSales = 0;
	int minSales = 9999999;

	int maxCustomers = 0;
	int minCustomers = 9999999;

	// Transforms
	for (int i = 0; i < entries.size(); i++) {
		if (stores.find(entries[i]->_store) == stores.end())
			stores.insert(entries[i]->_store);

		maxSales = std::max(maxSales, entries[i]->_sales);
		minSales = std::min(minSales, entries[i]->_sales);

		maxCustomers = std::max(maxCustomers, entries[i]->_customers);
		minCustomers = std::min(minCustomers, entries[i]->_customers);
	}

	const int valuesPerStore = 12;

	int numInputs = valuesPerStore * stores.size();

	int dim = static_cast<int>(std::ceil(std::sqrt(static_cast<float>(numInputs))));
	
	// Create model
	std::vector<neo::PredictiveHierarchy::LayerDesc> layerDescs(3);

	layerDescs[0]._width = 32;
	layerDescs[0]._height = 32;

	layerDescs[1]._width = 24;
	layerDescs[1]._height = 24;

	layerDescs[2]._width = 16;
	layerDescs[2]._height = 16;

	neo::PredictiveHierarchy ph;

	ph.createRandom(dim, dim, 16, layerDescs, -0.01f, 0.01f, 0.01f, 0.05f, 0.1f, generator);

	for (int it = 0; it < 4; it++) {
		// Go through series one day at a time
		for (int i = 0; i < entries.size(); i += stores.size()) {
			int inde
			int inputIndex = 0;

			for (int s = 0; s < stores.size(); s++) {
				ph.setInput(inputIndex++, entries[)
			}
			// If is test data, run off of own predictions with learning turned off

			// For each store
			
		}
	}

	return 0;
}

#endif
