package main

import (
	"log"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyconv"
	"github.com/unixpickle/anynet/anyff"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/mnist"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/slidingrbf"
)

var Creator anyvec.Creator

func main() {
	log.Println("Setting up...")

	Creator = anyvec32.CurrentCreator()

	network := anynet.Net{
		slidingrbf.NewLayer(Creator, 28, 28, 1, 3, 3, 8, 2, 2),
		anyconv.NewBatchNorm(Creator, 8),
		slidingrbf.NewLayer(Creator, 13, 13, 8, 4, 4, 8, 1, 1),
		anyconv.NewBatchNorm(Creator, 8),
		slidingrbf.NewLayer(Creator, 10, 10, 8, 3, 3, 16, 2, 2),
		anyconv.NewBatchNorm(Creator, 16),
		anynet.NewFC(Creator, 16*4*4, 10),
		anynet.LogSoftmax,
	}

	t := &anyff.Trainer{
		Net:     network,
		Cost:    anynet.DotCost{},
		Params:  network.Parameters(),
		Average: true,
	}

	var iterNum int
	s := &anysgd.SGD{
		Fetcher:     t,
		Gradienter:  t,
		Transformer: &anysgd.Adam{},
		Samples:     mnist.LoadTrainingDataSet().AnyNetSamples(Creator),
		Rater:       anysgd.ConstRater(0.001),
		StatusFunc: func(b anysgd.Batch) {
			log.Printf("iter %d: cost=%v", iterNum, t.LastCost)
			iterNum++
		},
		BatchSize: 200,
	}

	log.Println("Press ctrl+c once to stop...")
	s.Run(rip.NewRIP().Chan())

	log.Println("Post training...")
	pt := &anyconv.PostTrainer{
		Samples:   s.Samples,
		Fetcher:   t,
		BatchSize: 300,
		Net:       network,
	}
	pt.Run()

	log.Println("Computing statistics...")
	printStats(network)
}

func printStats(net anynet.Net) {
	ts := mnist.LoadTestingDataSet()
	cf := func(in []float64) int {
		vec := Creator.MakeVectorData(Creator.MakeNumericList(in))
		inRes := anydiff.NewConst(vec)
		res := net.Apply(inRes, 1).Output()
		return anyvec.MaxIndex(res)
	}
	log.Println("Validation:", ts.NumCorrect(cf))
	log.Println("Histogram:", ts.CorrectnessHistogram(cf))
}
