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

class VectorCodec {
private:
	std::unordered_set<char> alphabet;
	char tableStoI[256];
	char tableItoS[256];
public:
	int N;
	int nSymbols;
	std::vector<float> vector;
	char symbol;
	int symIndex;
	
	VectorCodec(std::string corpus, int vecLength = 0) {
		for (int i = 0; i < corpus.length(); i++) {
			alphabet.emplace(corpus[i]);
		}
		nSymbols = alphabet.size();
				
		if (vecLength == 0) { //auto
			N = nSymbols;
		} else {
			N = vecLength;
		}
		vector.resize(N, 0.0f);
		
		for (int i = 0; i < 256; i++) { tableStoI[i] = 0; tableItoS[i] = 0; }
		
		int index = 0;
		for (auto itr = alphabet.begin(); itr != alphabet.end(); ++itr) {
			tableStoI[*itr] = index;
			tableItoS[index] = *itr;
			index++;
		}
		
		symbol = 0;
		symIndex = 0;
	}
	
	void encode() {
		for (int i = 0; i < vector.size(); i++) {
			vector[i] = 0.0f;
		}
		vector[tableStoI[symbol]] = 1.0f;
	};
	
	void decode() {
		int maxIndex = 0;
		for (int i = 0; i < N; i++) {
			if (vector[i] > vector[maxIndex])
				maxIndex = i;
		}
		symbol = tableItoS[maxIndex];
		symIndex = maxIndex;
	};
	
	char getRandomSymbol(std::mt19937& generator) {
		return tableItoS[generator() % nSymbols];
	}
};

void train(neo::PredictiveHierarchy& ph, std::mt19937& generator,
           std::string& test, int epochs, VectorCodec& textcodec) {
	
    for (size_t k = 0; k < epochs; k++) {
        for (size_t i = 0; i < test.length(); i++) {
            textcodec.symbol = test[i];
            textcodec.encode();
            
			for (int j = 0; j < textcodec.N; j++) {
                ph.setInput(j, textcodec.vector[j]);
            }

            ph.simStep(generator);
			
			for (int j = 0; j < textcodec.N; j++) {
                textcodec.vector[j] = ph.getPrediction(j);
            }
			
			textcodec.decode();
			
            char predChar = textcodec.symbol;

            std::cout << predChar;
        }
        std::cout << "\n";
    }
}

void sample(neo::PredictiveHierarchy& ph, std::mt19937& generator,
           char seed, int nSamples, VectorCodec& textcodec) {
    
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	
	textcodec.symbol = seed;
	textcodec.encode();
	
	for (int j = 0; j < textcodec.N; j++) {
        ph.setInput(j, textcodec.vector[j] + dist01(generator)*0.5);
    }
    ph.simStep(generator, false);
    
    std::cout << "seed: " << seed << " i: " << textcodec.symIndex << " sample: ";
    
    for (size_t i = 1; i < nSamples; i++) {

		for (int j = 0; j < textcodec.N; j++) {
			textcodec.vector[j] = ph.getPrediction(j);
		}
        
		textcodec.decode();
			
		char predChar = textcodec.symbol;
        
		std::cout << predChar << " ";
		
        for (int j = 0; j < textcodec.N; j++) {
            ph.setInput(j, ph.getPrediction(j) + dist01(generator)*0.05);
        }
        
        ph.simStep(generator, false);
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
	
	// ---------------------------------- Find Character Set ----------------------------------
	
	VectorCodec textcodec(test);
	int numInputs = textcodec.N;
	int inputsRoot = std::ceil(std::sqrt(static_cast<float>(numInputs)));
	
	// ---------------------------------- Create Hierarchy ----------------------------------
	
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
    std::cout << "Corpus: " << corpusPath << " size: " << test.length() << " alphabet size: " << textcodec.nSymbols << std::endl;
    std::cout << "Model: nLayers: " << nLayers << " layerW: " << layerW << " layerH: " << layerH << " inFeedbackRadius: " << inFeedBackRadius 
			  << " input: " << inputsRoot << "x" << inputsRoot << std::endl;
    
    train(ph, generator, test, numEpochs, textcodec);
    
    for (int i = 0; i < numSamples; i++) {
        sample(ph, generator, textcodec.getRandomSymbol(generator), test.length(), textcodec);
    }
    
	return 0;
}

#endif
