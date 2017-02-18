package slidingrbf

import (
	"reflect"
	"testing"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/serializer"
)

func TestSerializer(t *testing.T) {
	c := anyvec32.DefaultCreator{}
	net := NewLayer(c, 4, 3, 5, 2, 3, 2, 1, 4)
	data, err := serializer.SerializeAny(net)
	if err != nil {
		t.Fatal(err)
	}

	var net1 anynet.Layer
	if err := serializer.DeserializeAny(data, &net1); err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(net, net1) {
		t.Error("got unexpected value")
	}
}
