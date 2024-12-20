import json
import re
import subprocess
import os.path
import os
from tqdm import tqdm 

from .cpg_client_wrapper import CPGClientWrapper
from ..data import datamanager as data


def funcs_to_graphs(funcs_path):
    client = CPGClientWrapper()
    # query the cpg for the dataset
    print(f"Creating CPG.")
    graphs_string = client(funcs_path)
    # removes unnecessary namespace for object references
    graphs_string = re.sub(r"io\.shiftleft\.codepropertygraph\.generated\.", '', graphs_string)
    graphs_json = json.loads(graphs_string)

    return graphs_json["functions"]


def graph_indexing(graph):
    idx = int(graph["file"].split(".c")[0].split("/")[-1])
    del graph["file"]
    return idx, {"functions": [graph]}


##################################################################################
from concurrent.futures import ThreadPoolExecutor
import os
import shutil

def joern_parse_single(joern_path, input_path, output_path, file_name):
    out_file = file_name + ".bin"
    subprocess.run(
        ["./" + joern_path + "joern-parse", input_path, "--out", os.path.join(output_path, out_file)],
        stdout=subprocess.PIPE,
        text=True,
        check=True
    )
    return out_file

def joern_parse_parallel(context, slices, PATHS, cpg_path, FILES):
    def process_slice(slice_data):
        s, slice = slice_data
        data.to_files(slice, PATHS.joern)  # Assuming `data.to_files` is thread-safe
        cpg_file = joern_parse_single(
            context.joern_cli_dir, 
            PATHS.joern, 
            cpg_path, 
            f"{s}_{FILES.cpg}"
        )
        print(f"Dataset {s} to cpg.")
        shutil.rmtree(PATHS.joern)  # Clear temporary files
        return cpg_file

    cpg_files = []
    with ThreadPoolExecutor() as executor:
        for cpg_file in tqdm(executor.map(process_slice, slices), total=len(slices), desc="Processing slices"):
            cpg_files.append(cpg_file)

    return cpg_files

##################################################################################

def joern_parse(joern_path, input_path, output_path, file_name):
    out_file = file_name + ".bin"
    joern_parse_call = subprocess.run(["./" + joern_path + "joern-parse", input_path, "--out", os.path.join(output_path,out_file)],
                                      stdout=subprocess.PIPE, text=True, check=True)
    print(str(joern_parse_call))
    return out_file


def joern_create(joern_path, in_path, out_path, cpg_files):
    joern_process = subprocess.Popen(
        ["./" + joern_path + "joern"], 
        stdin=subprocess.PIPE, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True  # Use text mode for easier string handling
    )
    json_files = []
    for cpg_file in cpg_files:
        json_file_name = f"{os.path.splitext(cpg_file)[0]}.json"
        json_files.append(json_file_name)
        
        if os.path.exists(os.path.join(in_path, cpg_file)):
            json_out = os.path.join(os.path.abspath(out_path), json_file_name) # os.path.abspath(out_path)
            if os.path.exists(json_out):
                continue
            import_cpg_cmd = f'importCpg("{os.path.abspath(in_path)}/{cpg_file}")\n'
            script_path = os.path.join(os.path.dirname(os.path.abspath(joern_path)), "graph-for-funcs.sc")
            run_script_cmd = f'cpg.runScript("{script_path}").toString() |> "{json_out}"\n'
            
            # Send import command
            joern_process.stdin.write(import_cpg_cmd)
            joern_process.stdin.flush()
            print("Import command output:", joern_process.stdout.readline().strip())
            
            # Send run script command
            joern_process.stdin.write(run_script_cmd)
            joern_process.stdin.flush()
            print("Run script command output:", joern_process.stdout.readline().strip())
            
            # Delete command
            joern_process.stdin.write("delete\n")
            joern_process.stdin.flush()
            print("Delete command output:", joern_process.stdout.readline().strip())
    
    try:
        outs, errs = joern_process.communicate(timeout=60)
        if outs:
            print("Final output:", outs)
        if errs:
            print("Errors:", errs)
    except subprocess.TimeoutExpired:
        joern_process.kill()
        outs, errs = joern_process.communicate()
        print("Process timed out and was killed.")
        if outs:
            print("Partial output:", outs)
        if errs:
            print("Partial errors:", errs)
    
    return json_files


def json_process(in_path, json_file):
    jsonpath = os.path.join(in_path, json_file)
    if os.path.exists(jsonpath):
        with open(jsonpath) as jf:
            cpg_string = jf.read()
            cpg_string = re.sub(r"io\.shiftleft\.codepropertygraph\.generated\.", '', cpg_string)
            cpg_json = json.loads(cpg_string)
            container = [graph_indexing(graph) for graph in cpg_json["functions"] if graph["file"] != "N/A"]
            return container
    return None

'''
def generate(dataset, funcs_path):
    dataset_size = len(dataset)
    print("Size: ", dataset_size)
    graphs = funcs_to_graphs(funcs_path[2:])
    print(f"Processing CPG.")
    container = [graph_indexing(graph) for graph in graphs["functions"] if graph["file"] != "N/A"]
    graph_dataset = data.create_with_index(container, ["Index", "cpg"])
    print(f"Dataset processed.")

    return data.inner_join_by_index(dataset, graph_dataset)
'''

# client = CPGClientWrapper()
# client.create_cpg("../../data/joern/")
# joern_parse("../../joern/joern-cli/", "../../data/joern/", "../../joern/joern-cli/", "gen_test")
# print(funcs_to_graphs("/data/joern/"))
"""
while True:
    raw = input("query: ")
    response = client.query(raw)
    print(response)
"""
