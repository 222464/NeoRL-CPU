#include "Settings.h"

#if EXAMPLE_SELECTION == EXAMPLE_TEXT_PREDICTION

#include <neo/PredictiveHierarchy.h>

#include "../libs/argparse.hpp"

#include <time.h>
#include <iostream>
#include <random>
#include <fstream>

#include <unordered_map>
#include <unordered_set>

#include <algorithm>

void train(neo::PredictiveHierarchy ph, std::mt19937 generator,
           std::string& test, int epochs, int numInputs, int inputsRoot,
           int minimum, int maximum) {
	
    for (size_t k = 0; k < epochs; k++) {
        for (size_t i = 0; i < test.length(); i++) {
            for (int j = 0; j < inputsRoot * inputsRoot; j++) {
                ph.setInput(j, 0.0f);
            }
            
            int index = test[i] - minimum;

            ph.setInput(index, 1.0f);

            ph.simStep(generator);

            int predIndex = 0;
            for (int i = 0; i < numInputs; i++) {
                if (ph.getPrediction(i) > ph.getPrediction(predIndex))
                    predIndex = i;
            }
			
            char predChar = predIndex + minimum;

            std::cout << predChar;
        }
        std::cout << "\n";
    }
}

void sample(neo::PredictiveHierarchy ph, std::mt19937 generator,
           char seed, int nSamples, int numInputs, int inputsRoot,
           int minimum, int maximum) {
    
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    for (int j = 0; j < inputsRoot * inputsRoot; j++) {
        ph.setInput(j, 0.0f);
    }
    ph.setInput(seed - minimum, 1.0f);//dist01(generator)*0.1+0.9);
    ph.simStep(generator);
    
    std::cout << "seed: " << seed << " sample: ";
    
    for (size_t i = 1; i < nSamples; i++) {
        
        ph.simStep(generator);

        int predIndex = 0;
        for (int i = 0; i < numInputs; i++) {
            if (ph.getPrediction(i) > ph.getPrediction(predIndex))
                predIndex = i;
        }
        
        char predChar = predIndex + minimum;
        std::cout << predIndex << " " << predChar << " ";
		
        for (int j = 0; j < inputsRoot * inputsRoot; j++) {
            ph.setInput(j, ph.getPrediction(j));
        }
    }
    
    std::cout << std::endl;
}

int main(int argc, const char** argv) {
	
    // Load the command line config
    
    ArgumentParser parser;
        
    parser.addArgument("-e", "--epochs", 1);
    parser.addArgument("-s", "--seed", 1);
    parser.addArgument("-l", "--layers", 1);
    parser.addArgument("-S", "--samples", 1);
    parser.addArgument("-c", "--corpus", 1);
    parser.addArgument("--nlayers", 1);
    parser.addArgument("--ifbradius", 1);
    parser.addArgument("--lw", 1);
    parser.addArgument("--lh", 1);
    
    parser.parse(argc, argv);
    
	// RNG
    unsigned int seed = std::atoi(parser.retrieve("seed", std::to_string(time(nullptr))).c_str());
	std::mt19937 generator(seed);

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    
	// ---------------------------------- Load the Corpus ----------------------------------
	std::string corpusPath = parser.retrieve("corpus", "corpus.txt");
	std::ifstream fromFile(corpusPath);

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
	
	// Fill out layer descriptions
	int nLayers = std::atoi(parser.retrieve("nlayers", "3").c_str());
	int layerW = std::atoi(parser.retrieve("lw", "16").c_str());
	int layerH = std::atoi(parser.retrieve("lh", "16").c_str());
	int inFeedBackRadius = std::atoi(parser.retrieve("ifbradius", "16").c_str());
	
	std::vector<neo::PredictiveHierarchy::LayerDesc> layerDescs(nLayers);
	
	for (int i = 0; i < nLayers; i++) {
		layerDescs[i]._width = layerW;
		layerDescs[i]._height = layerH;
	}
    
	neo::PredictiveHierarchy ph;

	ph.createRandom(inputsRoot, inputsRoot, inFeedBackRadius, layerDescs, -0.01f, 0.01f, 0.01f, 0.05f, 0.1f, generator);

	// ---------------------------------- Iterate Over Corpus ----------------------------------
    int numEpochs = std::atoi(parser.retrieve("epochs", "10").c_str());
    int numSamples = std::atoi(parser.retrieve("samples", "10").c_str());
    
    std::cout << "NeoRL text prediction experiment" << std::endl;
    std::cout << "Corpus: " << corpusPath << " size: " << test.length() << " minChar: " << minimum << " maxChar: " << maximum << std::endl;
    std::cout << "Model: nLayers: " << nLayers << " layerW: " << layerW << " layerH: " << layerH << " inFeedbackRadius: " << inFeedBackRadius << std::endl;
    
    train(ph, generator, test, numEpochs, numInputs, inputsRoot, minimum, maximum);
    
    for (int i = 0; i < numSamples; i++) {
        sample(ph, generator, 'I', test.length(), numInputs, inputsRoot, minimum, maximum);
    }
    
	return 0;
}

#endif
