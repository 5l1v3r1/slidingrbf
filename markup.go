package slidingrbf

import (
	"errors"
	"fmt"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyconv"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/convmarkup"
)

// FromMarkup creates a network from markup, as defined in
// https://github.com/unixpickle/convmarkup.
// The markup language is extended to include a SlidingRBF
// block, as noted in MarkupCreators().
func FromMarkup(c anyvec.Creator, code string) (anynet.Layer, error) {
	parsed, err := convmarkup.Parse(code)
	if err != nil {
		return nil, errors.New("parse markup: " + err.Error())
	}
	block, err := parsed.Block(convmarkup.Dims{}, MarkupCreators())
	if err != nil {
		return nil, errors.New("make markup block: " + err.Error())
	}
	chain := convmarkup.RealizerChain{
		convmarkup.MetaRealizer{},
		anyconv.Realizer(c),
		Realizer(c),
	}
	instance, _, err := chain.Realize(convmarkup.Dims{}, block)
	if err != nil {
		return nil, errors.New("realize markup block: " + err.Error())
	}
	if layer, ok := instance.(anynet.Layer); ok {
		return layer, nil
	} else {
		return nil, fmt.Errorf("not an anynet.Layer: %T", instance)
	}
}

// Realizer produces a convmarkup.Realizer for creating
// sliding RBF layers.
// This is intended to be used in a chain after a realizer
// from anyconv.
func Realizer(c anyvec.Creator) convmarkup.Realizer {
	return &realizer{creator: c}
}

// MarkupCreators returns a list of creators for parsing
// using convmarkup.
// This adds on to the default convmarkup creators.
//
// A single block is added: SlidingRBF.
// The SlidingRBF block takes the same parameters as a
// Conv block.
func MarkupCreators() map[string]convmarkup.Creator {
	res := convmarkup.DefaultCreators()
	res["SlidingRBF"] = createMarkupBlock
	return res
}

type realizer struct {
	creator anyvec.Creator
}

func (r *realizer) Realize(ch convmarkup.RealizerChain, d convmarkup.Dims,
	b convmarkup.Block) (interface{}, error) {
	mb, ok := b.(*markupBlock)
	if !ok {
		return nil, convmarkup.ErrUnsupportedBlock
	}
	return NewLayer(r.creator, d.Width, d.Height, d.Depth,
		mb.FilterWidth, mb.FilterHeight, mb.FilterCount,
		mb.StrideX, mb.StrideY), nil
}

type markupBlock struct {
	*convmarkup.Conv
}

func createMarkupBlock(in convmarkup.Dims, attr map[string]float64,
	children []convmarkup.Block) (convmarkup.Block, error) {
	conv, err := convmarkup.CreateConv(in, attr, children)
	if err != nil {
		return nil, err
	}
	return &markupBlock{conv.(*convmarkup.Conv)}, nil
}

func (m *markupBlock) Type() string {
	return "SlidingRBF"
}
