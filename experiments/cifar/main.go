package main

import (
	"flag"
	"log"
	"math/rand"
	"time"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyff"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/cifar"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/slidingrbf"
)

var Creator anyvec.Creator

func main() {
	Creator = anyvec32.CurrentCreator()
	rand.Seed(time.Now().UnixNano())

	var sampleDir string
	var netPath string
	var stepSize float64
	var batchSize int
	var successRate bool

	flag.StringVar(&sampleDir, "samples", "", "cifar-10 binary batch dir")
	flag.StringVar(&netPath, "net", "out_net", "network file")
	flag.Float64Var(&stepSize, "step", 0.001, "SGD step size")
	flag.IntVar(&batchSize, "batch", 64, "SGD batch size")
	flag.BoolVar(&successRate, "successrate", false, "print success rate")
	flag.Parse()

	if sampleDir == "" {
		essentials.Die("Missing -samples flag. See -help.")
	}

	lists, err := cifar.Load10(sampleDir)
	if err != nil {
		essentials.Die(err)
	}

	training := cifar.NewSampleListAll(Creator, lists[:5]...)
	validation := cifar.NewSampleListAll(Creator, lists[5])
	training.Augment = true

	var net anynet.Net
	if err := serializer.LoadAny(netPath, &net); err != nil {
		log.Println("Creating new network...")
		markup := `
			Input(w=32, h=32, d=3)
			SlidingRBF(w=3, h=3, n=8, sx=2, sy=2)
			BatchNorm
			SlidingRBF(w=4, h=4, n=8)
			BatchNorm
			SlidingRBF(w=3, h=3, n=16, sx=2, sy=2)
			BatchNorm
			FC(out=10)
			Softmax
		`
		layer, err := slidingrbf.FromMarkup(Creator, markup)
		if err != nil {
			essentials.Die(err)
		}
		net = layer.(anynet.Net)
	} else {
		log.Println("Using existing network.")
	}

	if successRate {
		log.Println("Computing success rate...")
		rate := validation.Accuracy(net, batchSize).(float32)
		log.Printf("Got %.3f%%", 100*rate)
	}

	log.Println("Setting up...")

	Creator = anyvec32.CurrentCreator()

	t := &anyff.Trainer{
		Net:     net,
		Cost:    anynet.DotCost{},
		Params:  net.Parameters(),
		Average: true,
	}

	var iterNum int
	s := &anysgd.SGD{
		Fetcher:     t,
		Gradienter:  t,
		Transformer: &anysgd.Adam{},
		Samples:     training,
		Rater:       anysgd.ConstRater(stepSize),
		StatusFunc: func(b anysgd.Batch) {
			anysgd.Shuffle(validation)
			batch, _ := t.Fetch(validation.Slice(0, batchSize))
			vCost := anyvec.Sum(t.TotalCost(batch.(*anyff.Batch)).Output())

			log.Printf("iter %d: cost=%v validation=%v", iterNum, t.LastCost, vCost)
			iterNum++
		},
		BatchSize: batchSize,
	}

	log.Println("Press ctrl+c once to stop...")
	s.Run(rip.NewRIP().Chan())

	log.Println("Saving network...")
	if err := serializer.SaveAny(netPath, net); err != nil {
		essentials.Die(err)
	}
}
