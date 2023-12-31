### preprocessor-hint: private-file
"""
A standalone script to test a counterexample. Requires the onnxruntime-gpu
package in a fresh python environment.

pip install onnx onnxruntime-gpu

First produce a counter example using --save_adv_example --cex_path flags:

cd complete_verifier/
python abcrown.py --config exp_configs/vnncomp22/collins-rul-cnn.yaml --onnx_path ../../vnncomp2022_benchmarks/benchmarks/collins_rul_cnn/onnx/NN_rul_full_window_40.onnx --vnnlib_path ../../vnncomp2022_benchmarks/benchmarks/collins_rul_cnn/vnnlib/robustness_16perturbations_delta40_epsilon10_w40.vnnlib --results_file out.txt --timeout 1200 --save_adv_example --cex_path counter_example.txt

Then check with this script, given the onnx file path, vnnlib file path and
counterexample path (counterexample is formatted according to VNN-COMP 2022
rules):

python check_counterexample.py ../../vnncomp2022_benchmarks/benchmarks/collins_rul_cnn/onnx/NN_rul_full_window_40.onnx ../../vnncomp2022_benchmarks/benchmarks/collins_rul_cnn/vnnlib/robustness_16perturbations_delta40_epsilon10_w40.vnnlib counter_example.txt

TODO: test more general specifications (it is tested on collins only).

"""

import re
import argparse
from copy import deepcopy
from collections import defaultdict
import numpy as np
import onnx
import onnx.numpy_helper
import onnxruntime

def parse_cex(cex_file):
    """Parse the saved counter example file."""
    x_dict = defaultdict(int)
    y_dict = defaultdict(int)
    max_x_dim = -1
    max_y_dim = -1
    reg_match = re.compile(r'\(+\s?([XY])_(\d+)\s+([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)\s?\)+')
    with open(cex_file) as f:
        for line in f:
            m = reg_match.match(line.strip())
            if m:
                xy, dim, val = m.group(1), m.group(2), m.group(3)
                dim = int(dim)
                val = float(val)
                if xy == 'X':
                    max_x_dim = max(max_x_dim, dim)
                    x_dict[dim] = val
                elif xy == 'Y':
                    max_y_dim = max(max_y_dim, dim)
                    y_dict[dim] = val
    max_x_dim += 1
    max_y_dim += 1
    print(f'Loaded input variables with dimension {max_x_dim} with {len(x_dict)} nonzeros.')
    print(f'Loaded output variables with dimension {max_y_dim} with {len(y_dict)} nonzeros.')
    x = np.zeros(max_x_dim)
    y = np.zeros(max_y_dim)
    for i in range(max_x_dim):
        x[i] = x_dict[i]
    for i in range(max_y_dim):
        y[i] = y_dict[i]
    return x, y

def check_vnnlib(vnnlib_file, x, tolerance=1e-4):
    """Check if x (input) is within range of input spec."""
    clipped_x = x.copy().astype(np.float64)
    vnnlib = read_vnnlib_simple(vnnlib_file)
    spec_x = vnnlib[0][0]
    # TODO: support more general specs. Currently only works for collins.
    spec_c = vnnlib[0][1][0][0]
    spec_y = vnnlib[0][1][0][1]
    max_violation = 0.
    for i, (x_l, x_u) in enumerate(spec_x):
        # print(f'input dim {i} counter example value {x[i]}, lower bound {x_l}, upper bound {x_u}')
        # Clipping for numerical errors.
        clipped_x[i] = min(max(x_l, clipped_x[i]), x_u)
        max_violation = max(max_violation, x_l - x[i])
        max_violation = max(max_violation, x[i] - x_u)
        assert x[i] >= x_l - tolerance
        assert x[i] <= x_u + tolerance
    print(f'max input violation = {max_violation} (should be <= {tolerance})')
    # TODO: check y for general properties.
    return clipped_x, spec_c, spec_y

def check_spec(y, spec_c, spec_y, tolerance=1e-4):
    """Check if the produced counterexample is a true violation."""
    max_violation = -float("inf")
    for c, sy in zip(spec_c, spec_y):
        violation = (y.dot(c) - sy).item()
        print(f'output is {y}, c is {c}, spec is {sy}, violation is {violation}')
        max_violation = max(violation, max_violation)
        assert violation <= tolerance  # Will fail if this is not an valid counter example.
    print(f'Counter example checking paseed. Max spec violation: {max_violation} (should be <= {tolerance})')

def convert_to_double(onnx_file):
    """Convert onnx file from single precision to double precision."""
    onnx_model = onnx.load(onnx_file)
    converted_names = []
    # Change all initializer to fp64.
    for i in range(len(onnx_model.graph.initializer)):
        fp64_initializer = onnx.numpy_helper.to_array(onnx_model.graph.initializer[i]).astype(np.float64)
        onnx_model.graph.initializer[i].data_type = onnx.TensorProto.DOUBLE
        onnx_model.graph.initializer[i].raw_data = onnx.numpy_helper.from_array(fp64_initializer).raw_data
        converted_names.append(onnx_model.graph.initializer[i].name)
    # Change all inputs to fp64.
    for i in range(len(onnx_model.graph.input)):
        onnx_model.graph.input[i].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    # Change all outputs to fp64.
    for i in range(len(onnx_model.graph.output)):
        onnx_model.graph.output[i].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    print('Model converted to double precision.')
    return onnx_model

def eval_onnx(onnx_file, x, y, precision=np.float64):
    """Evaluate a onnx file and compare to the expected output."""
    print(f'ONNX inference with precision {precision}')
    onnxruntime.set_default_logger_severity(3)
    # Workaround for onnx bug, see issue #150
    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    if precision == np.float64:
        converted_model = convert_to_double(onnx_file)
        ort_sess = onnxruntime.InferenceSession(converted_model.SerializeToString(), providers=['CUDAExecutionProvider'], sess_options=options)
    else:
        ort_sess = onnxruntime.InferenceSession(onnx_file, providers=['CUDAExecutionProvider'], sess_options=options)
    input_name = ort_sess.get_inputs()[0].name
    input_shape = ort_sess.get_inputs()[0].shape
    if input_shape[0] == 'BatchSize':
        input_shape[0] = 1
    x = x.reshape(*input_shape)
    outputs = ort_sess.run(None, {input_name: x.astype(precision)})[0]
    print(f'inference output: {outputs}')
    print(f'expected output: {y}')
    assert np.sum(np.abs(outputs - y)) <= 1e-4
    return outputs
            
def main():
    parser = argparse.ArgumentParser(description='Test a generated counterexample on a onnx file.')
    parser.add_argument('onnx_file', type=str, help='Path to onnx file.')
    parser.add_argument('vnnlib_file', type=str, help='Path to vnnlib file.')
    parser.add_argument('cex_file', type=str, help='Path to counter example file.')
    args = parser.parse_args()
    x, y = parse_cex(args.cex_file)
    print('Checking vnnlib input, making sure x is withtin range.')
    clipped_x, spec_c, spec_y = check_vnnlib(args.vnnlib_file, x)
    print('\nPrediction using original x')
    pred = eval_onnx(args.onnx_file, x, y, precision=np.float32)
    check_spec(pred, spec_c, spec_y)
    print('\nPrediction using x clipped to double precision bounds')
    check_vnnlib(args.vnnlib_file, clipped_x)
    pred = eval_onnx(args.onnx_file, clipped_x, y, precision=np.float32)
    check_spec(pred, spec_c, spec_y)
    print('\nPrediction using double precision and original x')
    pred_fp64 = eval_onnx(args.onnx_file, x, y, precision=np.float64)
    check_spec(pred_fp64, spec_c, spec_y)
    print('\nPrediction using double precision and x clipped to double precision bounds')
    pred_fp64 = eval_onnx(args.onnx_file, clipped_x, y, precision=np.float64)
    check_spec(pred_fp64, spec_c, spec_y)


## The following functions are from https://github.com/stanleybak/nnenum/blob/master/src/nnenum/vnnlib.py
# With a minor modification to automatically find the number of inputs and outputs.

'''
vnnlib simple utilities
Stanley Bak
June 2021
'''

def read_statements(vnnlib_filename):
    '''process vnnlib and return a list of strings (statements)

    useful to get rid of comments and blank lines and combine multi-line statements
    '''

    with open(vnnlib_filename, 'r') as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines]
    assert len(lines) > 0

    # combine lines if case a single command spans multiple lines
    open_parentheses = 0
    statements = []
    current_statement = ''
    
    for line in lines:
        comment_index = line.find(';')

        if comment_index != -1:
            line = line[:comment_index].rstrip()
        
        if not line:
            continue

        new_open = line.count('(')
        new_close = line.count(')')

        open_parentheses += new_open - new_close

        assert open_parentheses >= 0, "mismatched parenthesis in vnnlib file"

        # add space
        current_statement += ' ' if current_statement else ''
        current_statement += line

        if open_parentheses == 0:
            statements.append(current_statement)
            current_statement = ''

    if current_statement:
        statements.append(current_statement)

    # remove repeated whitespace characters
    statements = [" ".join(s.split()) for s in statements]

    # remove space after '('
    statements = [s.replace('( ', '(') for s in statements]

    # remove space after ')'
    statements = [s.replace(') ', ')') for s in statements]

    return statements

def update_rv_tuple(rv_tuple, op, first, second, num_inputs, num_outputs):
    'update tuple from rv in read_vnnlib_simple, with the passed in constraint "(op first second)"'
    
    if first.startswith("X_"):
        # Input constraints
        index = int(first[2:])

        assert not second.startswith("X") and not second.startswith("Y"), \
                                     f"input constraints must be box ({op} {first} {second})"
        assert 0 <= index < num_inputs

        limits = rv_tuple[0][index]
        
        if op == "<=":
            limits[1] = min(float(second), limits[1])
        else:
            limits[0] = max(float(second), limits[0])

        assert limits[0] <= limits[1], f"{first} range is empty: {limits}"

    else:
        # output constraint
        if op == ">=":
            # swap order if op is >=
            first, second = second, first

        row = [0.0] * num_outputs
        rhs = 0.0

        # assume op is <=
        if first.startswith("Y_") and second.startswith("Y_"):
            index1 = int(first[2:])
            index2 = int(second[2:])

            row[index1] = 1
            row[index2] = -1
        elif first.startswith("Y_"):
            index1 = int(first[2:])
            row[index1] = 1
            rhs = float(second)
        else:
            assert second.startswith("Y_")
            index2 = int(second[2:])
            row[index2] = -1
            rhs = -1 * float(first)

        mat, rhs_list = rv_tuple[1], rv_tuple[2]
        mat.append(row)
        rhs_list.append(rhs)

def make_input_box_dict(num_inputs):
    'make a dict for the input box'

    rv = {i: [-np.inf, np.inf] for i in range(num_inputs)}

    return rv

def get_io_nodes(onnx_model):
    'returns 3 -tuple: input node, output nodes, input dtype'

    # Workaround for onnx bug, see issue #150
    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    sess = ort.InferenceSession(onnx_model.SerializeToString(), sess_options=options)
    inputs = [i.name for i in sess.get_inputs()]
    assert len(inputs) == 1, f"expected single onnx network input, got: {inputs}"
    input_name = inputs[0]

    outputs = [o.name for o in sess.get_outputs()]
    assert len(outputs) == 1, f"expected single onnx network output, got: {outputs}"
    output_name = outputs[0]

    g = onnx_model.graph
    inp = [n for n in g.input if n.name == input_name][0]
    out = [n for n in g.output if n.name == output_name][0]

    input_type = g.input[0].type.tensor_type.elem_type

    assert input_type in [onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE]

    dtype = np.float32 if input_type == onnx.TensorProto.FLOAT else np.float64

    return inp, out, dtype

def get_num_inputs_outputs(onnx_filename):
    'get num inputs, num outputs, and input dtype of an onnx file'

    onnx_model = onnx.load(onnx_filename)
    inp, out, inp_dtype = get_io_nodes(onnx_model)
    
    inp_shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in inp.type.tensor_type.shape.dim)
    out_shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in out.type.tensor_type.shape.dim)

    num_inputs = 1
    num_outputs = 1

    for n in inp_shape:
        num_inputs *= n

    for n in out_shape:
        num_outputs *= n

    return num_inputs, num_outputs, inp_dtype

def read_vnnlib_simple(vnnlib_filename):
    '''process in a vnnlib file. You can get num_inputs and num_outputs using get_num_inputs_outputs().

    this is not a general parser, and assumes files are provided in a 'nice' format. Only a single disjunction
    is allowed

    output a list containing 2-tuples:
        1. input ranges (box), list of pairs for each input variable
        2. specification, provided as a list of pairs (mat, rhs), as in: mat * y <= rhs, where y is the output. 
                          Each element in the list is a term in a disjunction for the specification.
    '''

    # example: "(declare-const X_0 Real)"
    regex_declare = re.compile(r"^\(declare-const (X|Y)_(\S+) Real\)$")

    lines = read_statements(vnnlib_filename)

    # Read lines to determine number of inputs and outputs
    num_inputs = num_outputs = 0
    for line in lines:
        declare = regex_declare.findall(line)
        if len(declare) == 0:
            continue
        elif len(declare) > 1:
            raise ValueError(f'There cannot be more than one declaration in one line: {line}')
        else:
            declare = declare[0]
            if declare[0] == 'X':
                num_inputs = max(num_inputs, int(declare[1]) + 1)
            elif declare[0] == 'Y':
                num_outputs = max(num_outputs, int(declare[1]) + 1)
            else:
                raise ValueError(f'Unknown declaration: {line}')
    print(f'{num_inputs} inputs and {num_outputs} outputs in vnnlib')

    # comparison sub-expression
    # example: "(<= Y_0 Y_1)" or "(<= Y_0 10.5)"
    comparison_str = r"\((<=|>=) (\S+) (\S+)\)"

    # example: "(and (<= Y_0 Y_2)(<= Y_1 Y_2))"
    dnf_clause_str = r"\(and (" + comparison_str + r")+\)"
    
    # example: "(assert (<= Y_0 Y_1))"
    regex_simple_assert = re.compile(r"^\(assert " + comparison_str + r"\)$")

    # disjunctive-normal-form
    # (assert (or (and (<= Y_3 Y_0)(<= Y_3 Y_1)(<= Y_3 Y_2))(and (<= Y_4 Y_0)(<= Y_4 Y_1)(<= Y_4 Y_2))))
    regex_dnf = re.compile(r"^\(assert \(or (" + dnf_clause_str + r")+\)\)$")

    rv = [] # list of 3-tuples, (box-dict, mat, rhs)
    rv.append((make_input_box_dict(num_inputs), [], []))
    
    for line in lines:
        #print(f"Line: {line}")

        if len(regex_declare.findall(line)) > 0:
            continue

        groups = regex_simple_assert.findall(line)

        if groups:
            assert len(groups[0]) == 3, f"groups was {groups}: {line}"
            op, first, second = groups[0]

            for rv_tuple in rv:
                update_rv_tuple(rv_tuple, op, first, second, num_inputs, num_outputs)
                
            continue

        ################
        groups = regex_dnf.findall(line)
        assert groups, f"failed parsing line: {line}"
        
        tokens = line.replace("(", " ").replace(")", " ").split()
        tokens = tokens[2:] # skip 'assert' and 'or'

        conjuncts = " ".join(tokens).split("and")[1:]

        old_rv = rv
        rv = []

        for rv_tuple in old_rv:
            for c in conjuncts:
                rv_tuple_copy = deepcopy(rv_tuple)
                rv.append(rv_tuple_copy)
                
                c_tokens = [s for s in c.split(" ") if len(s) > 0]

                count = len(c_tokens) // 3

                for i in range(count):
                    op, first, second = c_tokens[3*i:3*(i+1)]

                    update_rv_tuple(rv_tuple_copy, op, first, second, num_inputs, num_outputs)

    # merge elements of rv with the same input spec
    merged_rv = {}

    for rv_tuple in rv:
        boxdict = rv_tuple[0]
        matrhs = (rv_tuple[1], rv_tuple[2])

        key = str(boxdict) # merge based on string representation of input box... accurate enough for now

        if key in merged_rv:
            merged_rv[key][1].append(matrhs)
        else:
            merged_rv[key] = (boxdict, [matrhs])

    # finalize objects (convert dicts to lists and lists to np.array)
    final_rv = []

    for rv_tuple in merged_rv.values():
        box_dict = rv_tuple[0]
        
        box = []

        for d in range(num_inputs):
            r = box_dict[d]

            assert r[0] != -np.inf and r[1] != np.inf, f"input X_{d} was unbounded: {r}"
            box.append(r)
            
        spec_list = []

        for matrhs in rv_tuple[1]:
            mat = np.array(matrhs[0], dtype=float)
            rhs = np.array(matrhs[1], dtype=float)
            spec_list.append((mat, rhs))

        final_rv.append((box, spec_list))

    #for i, (box, spec_list) in enumerate(final_rv):
    #    print(f"-----\n{i+1}. {box}\nspec:{spec_list}")
        
    return final_rv


if __name__ == "__main__":
    main()

