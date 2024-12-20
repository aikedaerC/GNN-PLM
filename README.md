# GNN-PLM

Implementation of GNN-PLM Model in Python with code for processing the dataset and generation of Code Property Graphs.

## Prerequisites

### Python

The experiments were executed on single NVIDIA A800 80GB GPU. The system specifications comprised NVIDIA driver version 535.183.01 and CUDA version 12.2.
```shell
conda create -n vul python==3.8
```

> copy the site-packages.tgz into `/home/user/anaconda3/envs/vul/lib/python3.8/`

The package can be downloaded from [google drive](https://drive.google.com/file/d/1YsVgUPciGOIqFNf9klWUQ9T2NT2g0cMv/view?usp=sharing)

### Java

jdk-8 or jdk-11 is ok.
```shell
sudo apt update
sudo apt install openjdk-11-jdk
vim ~/.bashrc
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk
export PATH=$JAVA_HOME/bin:$PATH
source ~/.bashrc
```

### Joern
Maybe I will keep this tools in the current joern folder for convenient, but I'm not sure whether it will be missing in the future. So the following will be focus on installing joern from scratch.

```shell
# download this version 
[Joern=v1.0.105](https://github.com/joernio/joern/releases/download/v1.0.105/joern-cli.zip)
# in the same folder run joern-install.sh
chmod u+x joern-install.sh
bash joern-install.sh --interactive
```


## Structure


## Usage

### Dataset

We collected 11 open-source function-level vulnerability detection datasets, which can be downloaded from [google drive](https://drive.google.com/file/d/1Z48QiXMIxrveZZYqypGNgCxEO18obC1Z/view?usp=sharing)

### baselines


### "main.py"

For each task, verify that the correct files are in the respective folders.
For example, executing the **Process** task requires the input datasets that contain 
the embedded graphs with the associated labels.

#### Create Task
This is the first task where the dataset is filtered (optionally) and augmented with a column that 
contains the respective Code Property Graph (CPG).

Functions in the dataset are written to files into a target directory which Joern is queried with for creating the CPG. 
After the CPG creation, Joern is queried with the script "graph-for-funcs.sc" which creates the graphs from the CPG.
Those are returned in JSON format, containing all the functions with the respective AST, CFG and PDG graphs.

Execute with:

``` console
python main.py -cpg
```

#### Embed Task
This task transforms the source code functions into tokens which are used to generate and train the word2vec model for the initial embeddings. The nodes embeddings are AST. 


Execute with:
``` 
python main.py -embed
```

#### Process Task

Execute with:
``` console
$ python main.py -p -path ckpts/xxx.pth
```


##### Tokenization example
Source code:
```
'static void v4l2_free_buffer(void *opaque, uint8_t *unused)
{

    V4L2Buffer* avbuf = opaque;

    V4L2m2mContext *s = buf_to_m2mctx(avbuf);



    if (atomic_fetch_sub(&avbuf->context_refcount, 1) == 1) {

        atomic_fetch_sub_explicit(&s->refcount, 1, memory_order_acq_rel);



        if (s->reinit) {

            if (!atomic_load(&s->refcount))

              sem_post(&s->refsync);

        } else if (avbuf->context->streamon)

            ff_v4l2_buffer_enqueue(avbuf);



        av_buffer_unref(&avbuf->context_ref);

    }

}
'
```
Tokens using tokenizer provided by Devign:
```
['static', 'void', 'FUN1', '(', 'void', '*', 'VAR1', ',', 'uint8_t', '*', 'VAR2)', '{', 'VAR3', '*', 'VAR4', '=', 'VAR1', ';', 'V4L2m2mContext', '*', 'VAR5', '=', 'FUN2', '(', 'VAR4)', ';', 'if', '(', 'FUN3', '(', '&', 'VAR4', '-', '>', 'VAR6', ',', '1)', '==', '1)', '{', 'FUN4', '(', '&', 'VAR5', '-', '>', 'VAR7', ',', '1', ',', 'VAR8)', ';', 'if', '(', 'VAR5', '-', '>', 'VAR9)', '{', 'if', '(', '!', 'FUN5', '(', '&', 'VAR5', '-', '>', 'VAR7))', 'FUN6', '(', '&', 'VAR5', '-', '>', 'VAR10)', ';', '}', 'else', 'if', '(', 'VAR4', '-', '>', 'VAR11', '-', '>', 'VAR12)', 'FUN7', '(', 'VAR4)', ';', 'FUN8', '(', '&', 'VAR4', '-', '>', 'VAR13)', ';', '}', '}']
```
 

## License
Distributed under the MIT License. See LICENSE for more information.

## Acknowledgments
Guidance and ideas for some parts from:

* [Transformer for Software Vulnerability Detection](https://github.com/hazimhanif/svd-transformer)
* [SySeVR: A Framework for Using Deep Learning to Detect Vulnerabilities](https://github.com/SySeVR/SySeVR)
* [VulDeePecker algorithm implemented in Python](https://github.com/johnb110/VDPython)
* [Devign](https://github.com/epicosy/devign)
* [Vul-LMGNN](https://github.com/Vul-LMGNN/vul-LMGGNN)
* [VulBERTa](https://github.com/ICL-ml4csec/VulBERTa)

