package slidingrbf

import (
	"math"
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anydifftest"
	"github.com/unixpickle/anyvec/anyvec64"
)

func TestDistLayerOut(t *testing.T) {
	layer := &DistLayer{
		InputWidth:   3,
		InputHeight:  3,
		InputDepth:   2,
		FilterWidth:  2,
		FilterHeight: 2,
		FilterCount:  2,
		StrideX:      1,
		StrideY:      1,
		Filters: anydiff.NewVar(anyvec64.MakeVectorData([]float64{
			// First filter
			1, 2, 3, 4,
			5, 6, 7, 8,
			// Second filter
			-1, -2, -3, -4,
			5, -6, 7, -8,
		})),
	}
	input := anydiff.NewConst(anyvec64.MakeVectorData([]float64{
		3, 2, 1, 2, 3, 2,
		3, 6, 1, 3, 5, -2,
		-3, 2, 0, 2, 4, 2,
	}))
	actual := layer.Apply(input, 1).Output().Data().([]float64)
	expected := []float64{77, 389, 133, 229, 190, 422, 127, 295}
	if len(actual) != len(expected) {
		t.Fatalf("expected len %d but got %d", len(expected), len(actual))
	}
	for i, x := range expected {
		a := actual[i]
		if math.Abs(x-a) > 1e-4 {
			t.Errorf("output %d: expected %f but got %f", i, x, a)
		}
	}
}

func TestDistLayerProp(t *testing.T) {
	layer := &DistLayer{
		InputWidth:   3,
		InputHeight:  3,
		InputDepth:   2,
		FilterWidth:  2,
		FilterHeight: 2,
		FilterCount:  2,
		StrideX:      1,
		StrideY:      1,
		Filters: anydiff.NewVar(anyvec64.MakeVectorData([]float64{
			// First filter
			1, 2, 3, 4,
			5, 6, 7, 8,
			// Second filter
			-1, -2, -3, -4,
			5, -6, 7, -8,
		})),
	}
	input := anydiff.NewVar(anyvec64.MakeVectorData([]float64{
		3, 2, 1, 2, 3, 2,
		3, 6, 1, 3, 5, -2,
		-3, 2, 0, 2, 4, 2,

		1, 2, 1, 2, 1, 2,
		3, 6, -1, 3, -1, -2,
		-3, 2, 1, -1, -4, -2,
	}))
	checker := anydifftest.ResChecker{
		F: func() anydiff.Res {
			return layer.Apply(input, 2)
		},
		V: []*anydiff.Var{input, layer.Filters},
	}
	checker.FullCheck(t)
}
