import os.path as osp
import os
import random
from typing import List
import shutil
from tqdm import tqdm
random.seed(2023)


def datacollect(dir:str) -> List["str"]:
    collect = []
    for i in os.listdir(dir):
        collect += datacollect(osp.join(dir,i)) if osp.isdir(osp.join(dir,i)) else [osp.join(dir,i),]
    return collect

def datacount(dir:str) -> int:
    counts = sum([datacount(osp.join(dir, i)) if osp.isdir(osp.join(dir,i)) else 1 for i in os.listdir(dir) ])
    return counts

class strategy():
    '''
    requiered to rewrite the prob method which should return a prob about each domain as a dict
    '''
    def __init__(self, domains:List[str], num_client:int, num_task:int ,num_sample_per:int) -> None:
        self.num_client = num_client
        self.num_task = num_task
        self.num_sample_per = num_sample_per
        self.domains = domains
    def prob(self, index:int) -> List[float]:
        raise NotImplementedError()
    
class single_strategy(strategy):
    def __init__(self, domains, num_client, num_task, num_sample_per) -> None:
        super().__init__(domains, num_client, num_task, num_sample_per)
    def prob(self, index: int) -> List[float]:
        cur_domain = self.domains[index%len(self.domains)]
        domain_prob = {i:0 for i in self.domains}
        domain_prob[cur_domain] = 1
        return domain_prob
        
class uniform_strategy(strategy):
    def __init__(self, domains, num_client, num_sample_per) -> None:
        super().__init__(domains, num_client, num_sample_per)
    def prob(self, index: int) -> List[float]:
        domain_prob = {i:1/len(self.domains) for i in self.domains}
        return domain_prob

class direchlet_strategy(strategy): # seems not indeed to do
    def __init__(self, domains, num_clients, num_sample_per, parameters) -> None:
        super().__init__(domains, num_clients, num_sample_per)
        self.dlt_parameters = parameters
    def prob(self, index):
        pass

def fcl_splitdata(indir:str,outdir:str, sample_strategy:strategy=None) :
    '''
    the indir should be the format like this:
    officehome/
        ├── Art
               ├── classes1
               ├── classes2
               ...
        ├── Clipart
        ├── Product
        └── Real World
    '''
    domain_data_collection = {i:datacollect(osp.join(indir, i)) for i in sample_strategy.domains}
    if os.path.exists(outdir):
        raise FileExistsError()
    else:
        os.makedirs(outdir)
    for i in range(sample_strategy.num_client): # calculate domain sample prob for each client and copy the file 
        print(f"client {i} sampling , strategy : {sample_strategy.__class__.__name__}")
        for task in range(sample_strategy.num_task):
            prob = sample_strategy.prob(task)
            num_per_domain = {domain:int(prob[domain]*sample_strategy.num_sample_per) for domain in prob} # get prob form strategy
            print(f"client {i} each domain prob: {num_per_domain}")
            cur_client_data = []
            for domain in num_per_domain:
                cur_client_data += random.sample(domain_data_collection[domain], num_per_domain[domain])
            for path in tqdm(cur_client_data):
                newdir = os.path.join(outdir, f"client{i}",f"task{task}", *(path.split(indir)[1].split(os.path.sep)[:-1])) # concat the new path
                # print(path, newdir)
                # exit()
                os.makedirs(newdir, exist_ok=True)
                newpath = os.path.join(newdir,path.split(os.path.sep)[-1])
                shutil.copy(path, newpath)




def time_seq_splitdata(indir:str,outdir:str, sample_strategy:strategy=None) :
    '''
    the indir should be the format like this:
    officehome/
        ├── Art
               ├── classes1
               ├── classes2
               ...
        ├── Clipart
        ├── Product
        └── Real World
    '''
    domain_data_collection = {i:datacollect(osp.join(indir, i)) for i in sample_strategy.domains}
    if os.path.exists(outdir):
        raise FileExistsError()
    else:
        os.makedirs(outdir)
    for i in range(sample_strategy.num_client): # calculate domain sample prob for each client and copy the file 
        print(f"client {i} sampling , strategy : {sample_strategy.__class__.__name__}")
        prob = sample_strategy.prob(i)
        num_per_domain = {domain:int(prob[domain]*sample_strategy.num_sample_per) for domain in prob} # get prob form strategy
        print(f"client {i} each domain prob: {num_per_domain}")
        cur_client_data = []
        for domain in num_per_domain:
            cur_client_data += random.sample(domain_data_collection[domain], num_per_domain[domain])
        for path in tqdm(cur_client_data):
            newdir = os.path.join(outdir, f"client{i}", *(path.split(os.path.sep)[1:-1])) # concat the new path
            os.makedirs(newdir, exist_ok=True)
            newpath = os.path.join(newdir,path.split(os.path.sep)[-1])
            shutil.copy(path, newpath)
            
        



if __name__ == "__main__":
    INDIR = "/mnt/sda/zd/data/officehome"
    OUTDIR = 'splitdata'
    sample_strategy = single_strategy(os.listdir(INDIR), 10, 4,1000)
    fcl_splitdata(INDIR, OUTDIR, sample_strategy)


    