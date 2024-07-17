package gonet

import (
	"log"
	"math"
	"runtime"
	"sync"
)

// NN struct is used to represent a neural network
type NN struct {
	// Whether the problem is regression or classification
	Regression bool
	// Number of nodes in each layer
	NNodes []int
	// Activations for each layer
	Activations [][]float64
	// Weights
	Weights [][][]float64
}

// New creates a new neural network
//
// 'nInputs' is number of nodes in input layer
//
// 'nHiddens' is array of numbers of nodes in hidden layers
//
// 'nOutputs' is number of nodes in output layer
//
// 'isRegression' is whether the problem is regression or classification
//
// return the neural network
func New(nInputs int, nHiddens []int, nOutputs int, isRegression bool) NN {
	nn := NN{}
	nn.Config(nInputs, nHiddens, nOutputs, isRegression)
	return nn
}

// Config the neural network, also reset all trained weights
//
// 'nInputs' is number of nodes in input layer
//
// 'nHiddens' is array of numbers of nodes in hidden layers
//
// 'nOutputs' is number of nodes in output layer
//
// 'isRegression' is whether the problem is regression or classification
func (nn *NN) Config(nInputs int, nHiddens []int, nOutputs int, isRegression bool) {
	if len(nHiddens) == 0 {
		log.Fatal("Should have at least 1 hidden layer")
	}

	nn.Regression = isRegression

	nn.NNodes = []int{
		nInputs + 1, // +1 for bias
	}
	for i := 0; i < len(nHiddens); i++ {
		nn.NNodes = append(nn.NNodes, nHiddens[i]+1) // +1 for bias
	}
	nn.NNodes = append(nn.NNodes, nOutputs)

	NLayers := len(nn.NNodes)

	nn.Activations = make([][]float64, 0)
	for i := 0; i < NLayers; i++ {
		nn.Activations = append(nn.Activations, vector(nn.NNodes[i], 1.0))
	}
	nn.Weights = make([][][]float64, NLayers-1)

	for i := 0; i < len(nn.Weights); i++ {
		nn.Weights[i] = matrix(nn.NNodes[i], nn.NNodes[i+1])
	}

	// rand.Seed(0)
	for i := 0; i < len(nn.Weights); i++ {
		for j := 0; j < len(nn.Weights[i]); j++ {
			for k := 0; k < len(nn.Weights[i][j]); k++ {
				nn.Weights[i][j][k] = random(-1, 1)
			}
		}
	}
}

// Feed forward the neural network
func (nn *NN) feedForward(inputs []float64) []float64 {
	NLayers := len(nn.NNodes)
	if NLayers < 3 {
		log.Fatal("Should have at least 1 hidden layer")
	}

	if len(inputs) != nn.NNodes[0]-1 {
		log.Fatal("Error: wrong number of inputs")
	}
	for i := 0; i < nn.NNodes[0]-1; i++ {
		nn.Activations[0][i] = inputs[i]
	}

	wg := sync.WaitGroup{}
	cpuCount := runtime.NumCPU()
	if cpuCount == 0 {
		cpuCount = 1
	}
	for k := 1; k < NLayers-1; k++ {
		istep := (nn.NNodes[k] - 1) / cpuCount
		if istep == 0 {
			istep = (nn.NNodes[k] - 1)
		} else {
			wg.Add(cpuCount)
		}
		if ((nn.NNodes[k] - 1) % cpuCount) != 0 {
			wg.Add(1)
		}
		for start := 0; start < nn.NNodes[k]-1; start += istep {
			end := start + istep
			if end > nn.NNodes[k]-1 {
				end = nn.NNodes[k] - 1
			}
			go func(k, istart, iend int) {
				defer wg.Done()
				for i := istart; i < iend; i++ {
					var sum float64

					for j := 0; j < nn.NNodes[k-1]; j++ {
						sum += nn.Activations[k-1][j] * nn.Weights[k-1][j][i]
					}

					if nn.Regression {
						// Use sigmoid to avoid explosion
						nn.Activations[k][i] = sigmoid(sum)
					} else {
						nn.Activations[k][i] = relu(sum)
					}
				}
			}(k, start, end)
		}
		wg.Wait()
	}

	istep := nn.NNodes[NLayers-1] / cpuCount
	if istep == 0 {
		istep = nn.NNodes[NLayers-1]
	} else {
		wg.Add(cpuCount)
	}
	if nn.NNodes[NLayers-1]%cpuCount != 0 {
		wg.Add(1)
	}
	for start := 0; start < nn.NNodes[NLayers-1]; start += istep {
		end := start + istep
		if end > nn.NNodes[NLayers-1] {
			end = nn.NNodes[NLayers-1]
		}
		go func(istart, iend int) {
			defer wg.Done()
			for i := istart; i < iend; i++ {
				var sum float64

				for j := 0; j < nn.NNodes[NLayers-2]; j++ {
					sum += nn.Activations[NLayers-2][j] * nn.Weights[NLayers-2][j][i]
				}

				if nn.Regression {
					nn.Activations[NLayers-1][i] = linear(sum)
				} else {
					nn.Activations[NLayers-1][i] = sigmoid(sum)
				}
			}
		}(start, end)
	}
	wg.Wait()

	return nn.Activations[NLayers-1]
}

// Update weights with Back Propagation algorithm
// 'targets' is traning outputs
// 'lRate' is learning rate
// 'mFactor' is used by momentum gradient discent
// return the prediction error
func (nn *NN) backPropagate(targets []float64, lRate float64) float64 {
	NLayers := len(nn.NNodes)
	if NLayers < 3 {
		log.Fatal("Should have at least 1 hidden layer")
	}

	if len(targets) != nn.NNodes[NLayers-1] {
		log.Fatal("Error: wrong number of target values")
	}

	deltas := make([][]float64, NLayers-1)
	deltas[NLayers-2] = vector(nn.NNodes[NLayers-1], 0.0)
	for node := 0; node < nn.NNodes[NLayers-1]; node++ {
		if nn.Regression {
			deltas[NLayers-2][node] = dlinear(nn.Activations[NLayers-1][node]) * (nn.Activations[NLayers-1][node] - targets[node])
		} else {
			deltas[NLayers-2][node] = dsigmoid(nn.Activations[NLayers-1][node]) * (nn.Activations[NLayers-1][node] - targets[node])
		}
	}

	wg := sync.WaitGroup{}
	for layer := len(deltas) - 2; layer >= 0; layer-- {
		deltas[layer] = vector(nn.NNodes[layer+1], 0.0)
		wg.Add(nn.NNodes[layer+1])
		for node := 0; node < nn.NNodes[layer+1]; node++ {
			go func(layer, node int) {
				defer wg.Done()
				var expect float64

				for nextNode := 0; nextNode < nn.NNodes[layer+2]-1; nextNode++ {
					expect += deltas[layer+1][nextNode] * nn.Weights[layer+1][node][nextNode] //
				}

				if nn.Regression {
					deltas[layer][node] = dsigmoid(nn.Activations[layer+1][node]) * expect
				} else {
					deltas[layer][node] = drelu(nn.Activations[layer+1][node]) * expect
				}
			}(layer, node)
		}
		wg.Wait()
	}

	for layer := NLayers - 2; layer >= 0; layer-- {
		wg.Add(nn.NNodes[layer])
		for node := 0; node < nn.NNodes[layer]; node++ {
			go func(layer, node int) {
				defer wg.Done()
				for nextLayerNode := 0; nextLayerNode < nn.NNodes[layer+1]; nextLayerNode++ {
					change := deltas[layer][nextLayerNode] * nn.Activations[layer][node]
					nn.Weights[layer][node][nextLayerNode] = nn.Weights[layer][node][nextLayerNode] - lRate*change
				}
			}(layer, node)
		}
		wg.Wait()
	}
	var err float64
	for i := 0; i < len(targets); i++ {
		err += 0.5 * math.Pow(targets[i]-nn.Activations[NLayers-1][i], 2)
	}
	return err
}

// Train the neural network
//
// 'inputs' is the training data
//
// 'iterations' is the number to run feed forward and back propagation
//
// 'lRate' is learning rate
//
// 'mFactor' is used by momentum gradient discent
//
// 'debug' is whether or not to log learning error every 1000 iterations
func (nn *NN) Train(inputs [][][]float64, iterations int, lRate float64, debug bool) {
	for i := 1; i <= iterations; i++ {
		var e float64
		for _, p := range inputs {
			nn.feedForward(p[0])

			tmp := nn.backPropagate(p[1], lRate)
			e += tmp
		}
		if debug && i%1000 == 0 {
			log.Printf("%d iterations: %f\n", i, e)
		}
	}
}

// Predict output with new input
func (nn *NN) Predict(input []float64) []float64 {
	return nn.feedForward(input)
}
