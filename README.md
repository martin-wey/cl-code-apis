# cl-code

## Data extraction
We extract a Python corpus from Github using the [Code-LMs](https://github.com/VHellendoorn/Code-LMs) repository. 
First, clone the repository, and navigate in the _Data_ folder. The full data extraction process is explained in the README.md file contained within this folder.
```sh
git clone https://github.com/VHellendoorn/Code-LMs
cd Code-LMs/Data
```
To obtain links to the Python repositories and extract those links, run:
```sh
mkdir TopLists
python3 gh_crawler.py python
```
Prior to that, replace the *TOKEN* in `gh_crawler.py` at line 6. To clone the repositories on your hardware, run:
```sh
cat TopLists/java-top-repos.txt | xargs -P16 -n1 -I% bash clone_repo.sh % java
```
And, to deduplicate the data, run:
```sh
python3 deduplicate.py
```

@todo: 
We want to extract codes that use specific versions of libraries. Therefore, we need to
- filter out projects without `requirements.txt`
- fetch cloned projects whose `requirements.txt` contains the version of the targetted library (e.g., `PyTorch>=1.7.0`)
- extract functions that use the library (not sure how to do that?)
