package gonet

import (
	"math"
	"math/rand"
	"runtime"
	"testing"
)

const testIterations = 5000

func TestPredict(t *testing.T) {
	cpus := runtime.NumCPU()
	t.Logf("Number of CPUs: %d", cpus)
	// AND test
	andTest := [][][]float64{
		{{0, 0}, {0}},
		{{0, 1}, {0}},
		{{1, 0}, {0}},
		{{1, 1}, {1}},
	}
	rand.Seed(0)
	nn := New(2, []int{cpus}, 1, false)
	nn.Train(andTest, testIterations, 0.4, false)
	for i := 0; i < len(andTest); i++ {
		input := andTest[i][0]
		output := andTest[i][1]
		predict := nn.Predict(input)
		if math.Round(predict[0]) != output[0] {
			t.Errorf("AND test failed. Got (%f AND %f) == %f, want %f.", input[0], input[1], math.Round(predict[0]), output[0])
		}
	}

	// OR test
	orTest := [][][]float64{
		{{0, 0}, {0}},
		{{0, 1}, {1}},
		{{1, 0}, {1}},
		{{1, 1}, {1}},
	}
	rand.Seed(0)
	nn.Config(2, []int{cpus}, 1, false)
	nn.Train(orTest, testIterations, 0.4, false)
	for i := 0; i < len(orTest); i++ {
		input := orTest[i][0]
		output := orTest[i][1]
		predict := nn.Predict(input)
		if math.Round(predict[0]) != output[0] {
			t.Errorf("OR test failed. Got (%f OR %f) == %f, want %f.", input[0], input[1], math.Round(predict[0]), output[0])
		}
	}

	// XOR test
	xorTest := [][][]float64{
		{{0, 0}, {0}},
		{{0, 1}, {1}},
		{{1, 0}, {1}},
		{{1, 1}, {0}},
	}
	rand.Seed(0)
	nn.Config(2, []int{cpus}, 1, false)
	nn.Train(xorTest, testIterations, 0.4, false)
	for i := 0; i < len(xorTest); i++ {
		input := xorTest[i][0]
		output := xorTest[i][1]
		predict := nn.Predict(input)
		if math.Round(predict[0]) != output[0] {
			t.Errorf("XOR test failed. Got (%f XOR %f) == %f, want %f.", input[0], input[1], math.Round(predict[0]), output[0])
		}
	}

	// Regression test
	regTest := [][][]float64{
		{{3}, {6}},
		{{4}, {8}},
		{{5}, {10}},
		{{6}, {12}},
	}
	rand.Seed(0)
	nn.Config(1, []int{cpus, cpus}, 1, true)
	nn.Train(regTest, testIterations, 0.6, false)
	t.Logf("model: %+v", nn)
	for i := 0; i < len(regTest); i++ {
		input := regTest[i][0]
		output := regTest[i][1]
		predict := nn.Predict(input)
		if math.Round(predict[0]) != output[0] {
			t.Errorf("Regression test failed. Got %f (rounded from %f) with input %f, want %f.", math.Round(predict[0]), predict[0], input[0], output[0])
		}
	}
}
