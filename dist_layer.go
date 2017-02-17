package slidingrbf

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet/anyconv"
	"github.com/unixpickle/anyvec"
)

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

// Apply applies the layer to a batch of input tensors.
func (d *DistLayer) Apply(in anydiff.Res, n int) anydiff.Res {
	if d.im2row == nil {
		d.initIm2Row()
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

		filterMat := &anyvec.Matrix{
			Data: d.Filters.Vector,
			Rows: d.FilterCount,
			Cols: d.Filters.Vector.Len() / d.FilterCount,
		}

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

func (d *DistLayer) initIm2Row() {
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
	// TODO: this.
}
