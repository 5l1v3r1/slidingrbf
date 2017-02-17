package slidingrbf

import (
	"math"

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
	scale := math.Sqrt(float64(filterX*filterY*inDepth) * 2)
	scaleVec := c.MakeVector(filterCount)
	scaleVec.AddScaler(c.MakeNumeric(scale))
	out := &rbfOutLayer{
		Scalers: anydiff.NewVar(scaleVec),
	}
	return anynet.Net{distLayer, out}
}

type rbfOutLayer struct {
	Scalers *anydiff.Var
}

func deserializeRBFOutLayer(d []byte) (*rbfOutLayer, error) {
	var s anyvecsave.S
	if err := serializer.DeserializeAny(d, &s); err != nil {
		return nil, err
	}
	return &rbfOutLayer{
		Scalers: anydiff.NewVar(s.Vector),
	}, nil
}

func (r *rbfOutLayer) Apply(in anydiff.Res, n int) anydiff.Res {
	c := in.Output().Creator()
	sq := anydiff.Pow(r.Scalers, c.MakeNumeric(-2))
	sq = anydiff.Scale(sq, c.MakeNumeric(-1))
	return anydiff.Exp(anydiff.ScaleRepeated(in, sq))
}

func (r *rbfOutLayer) SerializerType() string {
	return "github.com/unixpickle/slidingrbf.rbfOutLayer"
}

func (r *rbfOutLayer) Serialize() ([]byte, error) {
	return serializer.SerializeAny(&anyvecsave.S{Vector: r.Scalers.Vector})
}
