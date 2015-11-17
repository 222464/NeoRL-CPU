#include "Settings.h"

#if EXAMPLE_SELECTION == EXAMPLE_TEXT_PREDICTION

#include <neo/PredictiveHierarchy.h>

#include <time.h>
#include <iostream>
#include <random>
#include <fstream>

#include <unordered_map>
#include <unordered_set>

#include <algorithm>

int main() {
	// RNG
	std::mt19937 generator(time(nullptr));

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	// ---------------------------------- Load the Corpus ----------------------------------

	std::ifstream fromFile("corpus.txt");

	fromFile.seekg(0, std::ios::end);
	size_t size = fromFile.tellg();
	std::string test(size, ' ');
	fromFile.seekg(0);
	fromFile.read(&test[0], size);

	int minimum = 255;
	int maximum = 0;

	// ---------------------------------- Find Character Set ----------------------------------

	for (int i = 0; i < test.length(); i++) {
		minimum = std::min(static_cast<int>(test[i]), minimum);
		maximum = std::max(static_cast<int>(test[i]), maximum);
	}

	// ---------------------------------- Create Hierarchy ----------------------------------

	// Organize inputs into a square input region
	int numInputs = maximum - minimum + 1;

	int inputsRoot = std::ceil(std::sqrt(static_cast<float>(numInputs)));

	std::cout << "Dimensions: " << numInputs << " / " << inputsRoot << std::endl;
	std::cout << std::endl;

	std::vector<neo::PredictiveHierarchy::LayerDesc> layerDescs(3);

	// Fill out layer descriptions
	layerDescs[0]._width = 8;
	layerDescs[0]._height = 8;
	
	layerDescs[1]._width = 8;
	layerDescs[1]._height = 8;

	layerDescs[2]._width = 8;
	layerDescs[2]._height = 8;

	neo::PredictiveHierarchy ph;

	ph.createRandom(inputsRoot, inputsRoot, 8, layerDescs, -0.01f, 0.01f, 0.01f, 0.05f, 0.1f, generator);

	// ---------------------------------- Iterate Over Corpus ----------------------------------

	// Current character index
	int current = 0;

	for (size_t it = 0; it < 10000; it++) {
		for (int i = 0; i < inputsRoot * inputsRoot; i++)
			ph.setInput(i, 0.0f);

		int index = test[current] - minimum;

		ph.setInput(index, 1.0f);

		ph.simStep(generator);

		int predIndex = 0;

		for (int i = 1; i < inputsRoot * inputsRoot; i++)
			if (ph.getPrediction(i) > ph.getPrediction(predIndex))
				predIndex = i;

		char predChar = predIndex + minimum;

		std::cout << predChar;

		current = (current + 1) % test.length();

		float error = 1.0f;

		if (predChar == test[current])
			error = 0.0f;

		if (current == 0)
			std::cout << "\n";
	}

	std::cout << std::endl;

	std::cout << "Generating text:" << std::endl;
	std::cout << std::endl;

	char predChar = test[current];

	for (size_t it = 0; it < 1000; it++) {
		for (int i = 0; i < inputsRoot * inputsRoot; i++)
			ph.setInput(i, 0.0f);

		int index = predChar - minimum;

		ph.setInput(index, 1.0f);

		ph.simStepGenerate(generator, 0.01f);

		int predIndex = 0;

		for (int i = 1; i < inputsRoot * inputsRoot; i++)
			if (ph.getPrediction(i) > ph.getPrediction(predIndex))
				predIndex = i;

		predChar = predIndex + minimum;

		std::cout << predChar;
	}
	return 0;
}

#endif