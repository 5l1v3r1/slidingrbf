package slidingrbf

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet/anyconv"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvecsave"
	"github.com/unixpickle/serializer"
)

func init() {
	var d DistLayer
	serializer.RegisterTypedDeserializer(d.SerializerType(), DeserializeDistLayer)
}

// DistLayer computes square distances between the filters
// and their corresponding regions.
type DistLayer struct {
	InputWidth  int
	InputHeight int
	InputDepth  int

	FilterWidth  int
	FilterHeight int
	FilterCount  int

	StrideX int
	StrideY int

	Filters *anydiff.Var

	im2row *anyconv.Im2Row
}

// DeserializeDistLayer deserializes a DistLayer.
func DeserializeDistLayer(d []byte) (*DistLayer, error) {
	var filters *anyvecsave.S
	var inW, inH, inD, fW, fH, fC, sX, sY serializer.Int
	err := serializer.DeserializeAny(d, &inW, &inH, &inD, &fW, &fH, &fC, &sX, &sY, &filters)
	if err != nil {
		return nil, err
	}
	return &DistLayer{
		InputWidth:   int(inW),
		InputHeight:  int(inH),
		InputDepth:   int(inD),
		FilterWidth:  int(fW),
		FilterHeight: int(fH),
		FilterCount:  int(fC),
		StrideX:      int(sX),
		StrideY:      int(sY),
		Filters:      anydiff.NewVar(filters.Vector),
	}, nil
}

// OutputWidth returns the width of the output tensor.
func (d *DistLayer) OutputWidth() int {
	d.initIm2Row()
	return d.im2row.NumX()
}

// OutputWidth returns the height of the output tensor.
func (d *DistLayer) OutputHeight() int {
	d.initIm2Row()
	return d.im2row.NumY()
}

// OutputDepth returns the depth of the output tensor.
func (d *DistLayer) OutputDepth() int {
	d.initIm2Row()
	return d.FilterCount
}

// Apply applies the layer to a batch of input tensors.
func (d *DistLayer) Apply(in anydiff.Res, n int) anydiff.Res {
	d.initIm2Row()

	if d.im2row.InputSize()*n != in.Output().Len() {
		panic("invalid input length")
	}

	c := in.Output().Creator()

	var outputs []anyvec.Vector
	d.im2row.MapAll(in.Output(), func(_ int, m *anyvec.Matrix) {
		// This utilizes the observation that
		// (a-x)^2 = a^2 - 2ax + x^2.
		// By doing things that way, we can leverage a matrix
		// multiplication for -2ax.

		squared := m.Data.Copy()
		squared.Mul(m.Data)
		inSqSum := anyvec.SumCols(squared, m.Rows)
		filterSq := d.Filters.Output().Copy()
		filterSq.Mul(d.Filters.Output().Copy())
		filterSqSum := anyvec.SumCols(filterSq, d.FilterCount)

		filterMat := d.filterMat()

		product := &anyvec.Matrix{
			Data: c.MakeVector(m.Rows * d.FilterCount),
			Rows: m.Rows,
			Cols: d.FilterCount,
		}

		product.Product(false, true, c.MakeNumeric(1), m, filterMat, c.MakeNumeric(0))
		product.Data.Scale(c.MakeNumeric(-2))
		anyvec.AddChunks(product.Data, inSqSum)
		anyvec.AddRepeated(product.Data, filterSqSum)

		outputs = append(outputs, product.Data)
	})

	return &distLayerRes{
		Layer:  d,
		OutVec: c.Concat(outputs...),
		In:     in,
		V:      anydiff.MergeVarSets(anydiff.NewVarSet(d.Filters), in.Vars()),
	}
}

// Parameters returns the layer parameters.
func (d *DistLayer) Parameters() []*anydiff.Var {
	return []*anydiff.Var{d.Filters}
}

// SerializerType returns the unique ID used to serialize
// a DistLayer with the serializer package.
func (d *DistLayer) SerializerType() string {
	return "github.com/unixpickle/slidingrbf.DistLayer"
}

// Serialize serializes the layer.
func (d *DistLayer) Serialize() ([]byte, error) {
	return serializer.SerializeAny(
		serializer.Int(d.InputWidth),
		serializer.Int(d.InputHeight),
		serializer.Int(d.InputDepth),
		serializer.Int(d.FilterWidth),
		serializer.Int(d.FilterHeight),
		serializer.Int(d.FilterCount),
		serializer.Int(d.StrideX),
		serializer.Int(d.StrideY),
		&anyvecsave.S{Vector: d.Filters.Vector},
	)
}

func (d *DistLayer) initIm2Row() {
	if d.im2row != nil {
		return
	}
	d.im2row = &anyconv.Im2Row{
		InputWidth:   d.InputWidth,
		InputHeight:  d.InputHeight,
		InputDepth:   d.InputDepth,
		WindowWidth:  d.FilterWidth,
		WindowHeight: d.FilterHeight,
		StrideX:      d.StrideX,
		StrideY:      d.StrideY,
	}
}

func (d *DistLayer) filterMat() *anyvec.Matrix {
	return &anyvec.Matrix{
		Data: d.Filters.Vector,
		Rows: d.FilterCount,
		Cols: d.Filters.Vector.Len() / d.FilterCount,
	}
}

type distLayerRes struct {
	Layer  *DistLayer
	OutVec anyvec.Vector
	In     anydiff.Res
	V      anydiff.VarSet
}

func (d *distLayerRes) Output() anyvec.Vector {
	return d.OutVec
}

func (d *distLayerRes) Vars() anydiff.VarSet {
	return d.V
}

func (d *distLayerRes) Propagate(u anyvec.Vector, g anydiff.Grad) {
	c := u.Creator()
	inSize := d.Layer.im2row.InputSize()
	n := d.In.Output().Len() / inSize
	uChunkSize := u.Len() / n
	inIsVariable := g.Intersects(d.In.Vars())

	var downstream []anyvec.Vector
	d.Layer.im2row.MapAll(d.In.Output(), func(idx int, m *anyvec.Matrix) {
		uSlice := u.Slice(idx*uChunkSize, (idx+1)*uChunkSize)
		uMat := &anyvec.Matrix{
			Data: uSlice,
			Rows: d.Layer.OutputWidth() * d.Layer.OutputHeight(),
			Cols: d.Layer.OutputDepth(),
		}

		if filterGrad, ok := g[d.Layer.Filters]; ok {
			repeatedSum := anyvec.SumRows(uSlice, d.Layer.FilterCount)
			uGrad := d.Layer.Filters.Vector.Copy()
			anyvec.ScaleChunks(uGrad, repeatedSum)
			uGrad.Scale(c.MakeNumeric(2))
			filterGrad.Add(uGrad)

			filterMat := d.Layer.filterMat()
			filterMat.Data = filterGrad
			filterMat.Product(true, false, c.MakeNumeric(-2), uMat, m, c.MakeNumeric(1))
		}

		if inIsVariable {
			inUp := c.MakeVector(inSize)
			uSum := anyvec.SumCols(uSlice, uSlice.Len()/d.Layer.FilterCount)
			anyvec.ScaleChunks(m.Data, uSum)
			m.Product(false, false, c.MakeNumeric(-2), uMat, d.Layer.filterMat(),
				c.MakeNumeric(2))
			d.Layer.im2row.Mapper(c).MapTranspose(m.Data, inUp)
			downstream = append(downstream, inUp)
		}
	})

	if inIsVariable {
		d.In.Propagate(c.Concat(downstream...), g)
	}
}
