package slidingrbf

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvecsave"
	"github.com/unixpickle/serializer"
)

func init() {
	var r rbfOutLayer
	serializer.RegisterTypedDeserializer(r.SerializerType(), deserializeRBFOutLayer)
}

// NewLayer creates a full sliding-window radial basis
// function layer for the given input dimensions.
// The layer is initialized under the assumption that the
// inputs are statistically normalized.
func NewLayer(c anyvec.Creator, inWidth, inHeight, inDepth, filterX, filterY, filterCount,
	strideX, strideY int) anynet.Layer {
	filters := c.MakeVector(filterX * filterY * filterCount * inDepth)
	anyvec.Rand(filters, anyvec.Normal, nil)
	distLayer := &DistLayer{
		InputWidth:   inWidth,
		InputHeight:  inHeight,
		InputDepth:   inDepth,
		FilterWidth:  filterX,
		FilterHeight: filterY,
		FilterCount:  filterCount,
		StrideX:      strideX,
		StrideY:      strideY,
		Filters:      anydiff.NewVar(filters),
	}
	normalizer := 1 / (float64(filterX*filterY*inDepth) * 2)
	out := &rbfOutLayer{
		Normalizer: normalizer,
		Scalers:    anydiff.NewVar(c.MakeVector(filterCount)),
	}
	return anynet.Net{distLayer, out}
}

type rbfOutLayer struct {
	Normalizer float64
	Scalers    *anydiff.Var
}

func deserializeRBFOutLayer(d []byte) (*rbfOutLayer, error) {
	var s *anyvecsave.S
	var n serializer.Float64
	if err := serializer.DeserializeAny(d, &s, &n); err != nil {
		return nil, err
	}
	return &rbfOutLayer{
		Scalers:    anydiff.NewVar(s.Vector),
		Normalizer: float64(n),
	}, nil
}

func (r *rbfOutLayer) Apply(in anydiff.Res, n int) anydiff.Res {
	c := in.Output().Creator()
	scalers := anydiff.Exp(r.Scalers)
	scalers = anydiff.Scale(scalers, c.MakeNumeric(-r.Normalizer))
	return anydiff.Exp(anydiff.ScaleRepeated(in, scalers))
}

func (r *rbfOutLayer) Parameters() []*anydiff.Var {
	return []*anydiff.Var{r.Scalers}
}

func (r *rbfOutLayer) SerializerType() string {
	return "github.com/unixpickle/slidingrbf.rbfOutLayer"
}

func (r *rbfOutLayer) Serialize() ([]byte, error) {
	return serializer.SerializeAny(
		&anyvecsave.S{Vector: r.Scalers.Vector},
		serializer.Float64(r.Normalizer),
	)
}
