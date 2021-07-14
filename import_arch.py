import sys
import json

num_tabs = 0

def indent():
    global num_tabs
    num_tabs += 1

def unindent():
    global num_tabs
    num_tabs -= 1

def tprint(s):
    print ( "    "*num_tabs + s )


def print_layer_defs(layers):
    tprint( "def create_layers(self):" )
    indent()

    for layer in layers:
        if layer['type'] == 'dagnn.Conv':
            kern_h, kern_w, in_c, out_c = layer['block']['size']
            padding = layer['block']['pad'][0]
            stride = layer['block']['stride'][0]
            dilation = layer['block']['dilate'][0]
            tprint( f"self.{layer['name']} = nn.Conv2d({in_c}, {out_c}, ({kern_h},{kern_w}), stride={stride}, padding={padding}, dilation={dilation})" )
        elif layer['type'] == 'dagnn.ReLU':
            tprint( f"self.{layer['name']} = nn.ReLU()" )
        elif layer['type'] == 'dagnn.Pooling':
            kern_h, kern_w = layer['block']['poolSize']
            padding = layer['block']['pad']
            padding = (padding[2], padding[3], padding[0], padding[1])
            stride = tuple(layer['block']['stride'])
            if layer['block']['method'] == 'max':
                tprint( f"self.{layer['name']} = nn.Sequential( nn.ConstantPad2d({padding}, 0), nn.MaxPool2d( {(kern_h, kern_w)}, stride={stride}) )" )
        elif layer['type'] == 'dagnn.ConvTranspose':
            kern_h, kern_w, in_c, out_c = layer['block']['size']
            stride = tuple(layer['block']['upsample'])
            tprint( f"self.{layer['name']} = nn.ConvTranspose2d({in_c}, {out_c}, {(kern_h, kern_w)}, stride={stride}, bias=False)" )
    unindent()


def print_forward_pass(layers):
    tprint( f"def forward(self, init_corr_vol_norm):" )
    indent()

    in_vars = []
    out_var = ''

    for layer in layers:
        in_vars = layer['inputs']
        out_var = layer['outputs'][0]
        if layer['type'] != 'dagnn.Concat':
            tprint( f"{out_var} = self.{layer['name']}({in_vars[0]})" )
        else:
            in_vars_string = '(' + ','.join(in_vars) + ')'
            tprint( f"{out_var} = torch.cat( {in_vars_string}, dim=1 )" )

        if layer['type'] == 'dagnn.ConvTranspose':
            crop = layer['block']['crop']
            tprint( f"{out_var} = {out_var}[:,:,{crop[0]}:-{crop[1]},{crop[2]}:-{crop[3]}]" )

#    Debug - print sizes of feature vectors at each layer
#        tprint( f"print( ('{layer['name']}', '{out_var}', {out_var}.shape) )" )

    tprint( f"return {out_var}" )
    unindent()


def print_bumph():
    tprint( "import torch\nimport torch.nn as nn" )
    tprint( "\nclass WarpGenerator(nn.Module):" )
    indent()
    tprint( "def __init__(self):" )
    indent()
    tprint( "super(WarpGenerator,self).__init__()" )
    tprint( "self.create_layers()" )
    unindent()

with open( sys.argv[1], 'r' ) as json_file:
    layers = json.load( json_file )
    layers = [layer for layer in layers if layer['name'].startswith('init_geo')]
    for layer in layers:
        layer['name'] = layer['name'][5:]

    print_bumph()
    tprint("")
    print_layer_defs(layers)
    tprint("")
    print_forward_pass(layers)
