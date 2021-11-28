import argparse,json, os, re
import pandas as pd
# from plotly.offline import init_notebook_mode
# init_notebook_mode(connected=True) 
import nltk.corpus
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

from nltk.probability import _create_rand_fdist
from bertopic import BERTopic
import tensorflow as tf
import torch
# https://arxiv.org/help/api/user-manual
category_map = {'astro-ph': 'Astrophysics',
'astro-ph.CO': 'Cosmology and Nongalactic Astrophysics',
'astro-ph.EP': 'Earth and Planetary Astrophysics',
'astro-ph.GA': 'Astrophysics of Galaxies',
'astro-ph.HE': 'High Energy Astrophysical Phenomena',
'astro-ph.IM': 'Instrumentation and Methods for Astrophysics',
'astro-ph.SR': 'Solar and Stellar Astrophysics',
'cond-mat.dis-nn': 'Disordered Systems and Neural Networks',
'cond-mat.mes-hall': 'Mesoscale and Nanoscale Physics',
'cond-mat.mtrl-sci': 'Materials Science',
'cond-mat.other': 'Other Condensed Matter',
'cond-mat.quant-gas': 'Quantum Gases',
'cond-mat.soft': 'Soft Condensed Matter',
'cond-mat.stat-mech': 'Statistical Mechanics',
'cond-mat.str-el': 'Strongly Correlated Electrons',
'cond-mat.supr-con': 'Superconductivity',
'cs.AI': 'Artificial Intelligence',
'cs.AR': 'Hardware Architecture',
'cs.CC': 'Computational Complexity',
'cs.CE': 'Computational Engineering, Finance, and Science',
'cs.CG': 'Computational Geometry',
'cs.CL': 'Computation and Language',
'cs.CR': 'Cryptography and Security',
'cs.CV': 'Computer Vision and Pattern Recognition',
'cs.CY': 'Computers and Society',
'cs.DB': 'Databases',
'cs.DC': 'Distributed, Parallel, and Cluster Computing',
'cs.DL': 'Digital Libraries',
'cs.DM': 'Discrete Mathematics',
'cs.DS': 'Data Structures and Algorithms',
'cs.ET': 'Emerging Technologies',
'cs.FL': 'Formal Languages and Automata Theory',
'cs.GL': 'General Literature',
'cs.GR': 'Graphics',
'cs.GT': 'Computer Science and Game Theory',
'cs.HC': 'Human-Computer Interaction',
'cs.IR': 'Information Retrieval',
'cs.IT': 'Information Theory',
'cs.LG': 'Machine Learning',
'cs.LO': 'Logic in Computer Science',
'cs.MA': 'Multiagent Systems',
'cs.MM': 'Multimedia',
'cs.MS': 'Mathematical Software',
'cs.NA': 'Numerical Analysis',
'cs.NE': 'Neural and Evolutionary Computing',
'cs.NI': 'Networking and Internet Architecture',
'cs.OH': 'Other Computer Science',
'cs.OS': 'Operating Systems',
'cs.PF': 'Performance',
'cs.PL': 'Programming Languages',
'cs.RO': 'Robotics',
'cs.SC': 'Symbolic Computation',
'cs.SD': 'Sound',
'cs.SE': 'Software Engineering',
'cs.SI': 'Social and Information Networks',
'cs.SY': 'Systems and Control',
'econ.EM': 'Econometrics',
'eess.AS': 'Audio and Speech Processing',
'eess.IV': 'Image and Video Processing',
'eess.SP': 'Signal Processing',
'gr-qc': 'General Relativity and Quantum Cosmology',
'hep-ex': 'High Energy Physics - Experiment',
'hep-lat': 'High Energy Physics - Lattice',
'hep-ph': 'High Energy Physics - Phenomenology',
'hep-th': 'High Energy Physics - Theory',
'math.AC': 'Commutative Algebra',
'math.AG': 'Algebraic Geometry',
'math.AP': 'Analysis of PDEs',
'math.AT': 'Algebraic Topology',
'math.CA': 'Classical Analysis and ODEs',
'math.CO': 'Combinatorics',
'math.CT': 'Category Theory',
'math.CV': 'Complex Variables',
'math.DG': 'Differential Geometry',
'math.DS': 'Dynamical Systems',
'math.FA': 'Functional Analysis',
'math.GM': 'General Mathematics',
'math.GN': 'General Topology',
'math.GR': 'Group Theory',
'math.GT': 'Geometric Topology',
'math.HO': 'History and Overview',
'math.IT': 'Information Theory',
'math.KT': 'K-Theory and Homology',
'math.LO': 'Logic',
'math.MG': 'Metric Geometry',
'math.MP': 'Mathematical Physics',
'math.NA': 'Numerical Analysis',
'math.NT': 'Number Theory',
'math.OA': 'Operator Algebras',
'math.OC': 'Optimization and Control',
'math.PR': 'Probability',
'math.QA': 'Quantum Algebra',
'math.RA': 'Rings and Algebras',
'math.RT': 'Representation Theory',
'math.SG': 'Symplectic Geometry',
'math.SP': 'Spectral Theory',
'math.ST': 'Statistics Theory',
'math-ph': 'Mathematical Physics',
'nlin.AO': 'Adaptation and Self-Organizing Systems',
'nlin.CD': 'Chaotic Dynamics',
'nlin.CG': 'Cellular Automata and Lattice Gases',
'nlin.PS': 'Pattern Formation and Solitons',
'nlin.SI': 'Exactly Solvable and Integrable Systems',
'nucl-ex': 'Nuclear Experiment',
'nucl-th': 'Nuclear Theory',
'physics.acc-ph': 'Accelerator Physics',
'physics.ao-ph': 'Atmospheric and Oceanic Physics',
'physics.app-ph': 'Applied Physics',
'physics.atm-clus': 'Atomic and Molecular Clusters',
'physics.atom-ph': 'Atomic Physics',
'physics.bio-ph': 'Biological Physics',
'physics.chem-ph': 'Chemical Physics',
'physics.class-ph': 'Classical Physics',
'physics.comp-ph': 'Computational Physics',
'physics.data-an': 'Data Analysis, Statistics and Probability',
'physics.ed-ph': 'Physics Education',
'physics.flu-dyn': 'Fluid Dynamics',
'physics.gen-ph': 'General Physics',
'physics.geo-ph': 'Geophysics',
'physics.hist-ph': 'History and Philosophy of Physics',
'physics.ins-det': 'Instrumentation and Detectors',
'physics.med-ph': 'Medical Physics',
'physics.optics': 'Optics',
'physics.plasm-ph': 'Plasma Physics',
'physics.pop-ph': 'Popular Physics',
'physics.soc-ph': 'Physics and Society',
'physics.space-ph': 'Space Physics',
'q-bio.BM': 'Biomolecules',
'q-bio.CB': 'Cell Behavior',
'q-bio.GN': 'Genomics',
'q-bio.MN': 'Molecular Networks',
'q-bio.NC': 'Neurons and Cognition',
'q-bio.OT': 'Other Quantitative Biology',
'q-bio.PE': 'Populations and Evolution',
'q-bio.QM': 'Quantitative Methods',
'q-bio.SC': 'Subcellular Processes',
'q-bio.TO': 'Tissues and Organs',
'q-fin.CP': 'Computational Finance',
'q-fin.EC': 'Economics',
'q-fin.GN': 'General Finance',
'q-fin.MF': 'Mathematical Finance',
'q-fin.PM': 'Portfolio Management',
'q-fin.PR': 'Pricing of Securities',
'q-fin.RM': 'Risk Management',
'q-fin.ST': 'Statistical Finance',
'q-fin.TR': 'Trading and Market Microstructure',
'quant-ph': 'Quantum Physics',
'stat.AP': 'Applications',
'stat.CO': 'Computation',
'stat.ME': 'Methodology',
'stat.ML': 'Machine Learning',
'stat.OT': 'Other Statistics',
'stat.TH': 'Statistics Theory'}

def import_data(data_file):
    
    def get_metadata():
        with open(data_file, 'r') as f:
            for line in f:
                yield line
                
    titles = []
    abstracts = []
    years = []
    categories = []
    metadata = get_metadata()
    for paper in metadata:
        paper_dict = json.loads(paper)
        ref = paper_dict.get('journal-ref')
        try:
            year = int(ref[-4:]) 

            categories.append(category_map[paper_dict.get('categories').split(" ")[0]])
            years.append(year)
            titles.append(paper_dict.get('title'))
            abstracts.append(paper_dict.get('abstract'))
        except:
            pass 

    len(titles), len(abstracts), len(years), len(categories)
    import pandas as pd
    df = pd.DataFrame()

    df['titles'] = titles
    df['abstract']=abstracts
    df['years'] = years
    df['categories'] = categories
    print(df.head(5))
    return df


def select_data_by(df,yearRanges,catRanges):
    print('selecting data')
    DATAS=[]
    DATAStrings=[]
    if len(yearRanges)!=len(catRanges):
        print('years and categories need to match')
        return -1
    else:
        for years,cats in zip(yearRanges,catRanges):
            if years[0]=='all':
                if cats[0]=='all':
                    DATAStrings=['all_years-all_cats']
                    print('all years all cats selected')
                    return [df],DATAStrings
                else:
                    for cat in cats:
                        cat_df.append(df.loc[df['categories'] == cat])
                    print('all years - spec. cats selected')
                    DATAStrings.append('all_years-slctd_cats')
            else:
                cat_df=pd.DataFrame(),pd.DataFrame()
                df = df[(df['years'] >= years[0]) & (df['years'] <= years[1])]

                if cats[0]=='all':
                    DATAStrings.append(str(years[0])+'_'+str(years[1])+'-all_cats')
                    DATAS.append(df)
                    print('spec. years selected w all cats')
                else:
                    for cat in cats:
                        cat_df.append(df.loc[df['categories'] == cat])
                    DATAS.append(cat_df)
                    DATAStrings.append(str(years[0])+'_'+str(years[1])+'-') 
                    print('spec. years and specific cats')
    print(DATAS)
    print(DATAStrings)
    return DATAS,DATAStrings

def  clean_text(df, text_field, new_text_field_name):
    df[new_text_field_name] = df[text_field].str.lower()
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
    # remove numbers
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
    df[new_text_field_name] = df[new_text_field_name].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    return df


def train_BERTopic(text,strs):
    topicDF=pd.DataFrame()
    names,info,topicss,words,params,probs = [],[],[],[],[],[]
 

    topic_model = BERTopic(verbose=True, embedding_model="paraphrase-MiniLM-L12-v2",calculate_probabilities =False)
    topics, _ = topic_model.fit_transform(text)
    names.append(strs)
    info.append(topic_model.get_topic_info())
    topicss.append(topics)
    probs.append(_)
    params.append(topic_model.get_params())
    these=[]
    for t in topic_model.get_topic_info():
        these.append(t)
    words.append(these)
    print(topic_model.get_topic_info().head(10))
    txttpath='./results/info/'+strs
    imgpath ='./results/figs/'+strs
    if (os.path.isfile(txttpath)):
        topic_model.save('20-21_BertTopic')
    else:
        topic_model.save('_BertTopic')
        info.to_csv(txttpath+strs+'.csv')
    if (os.path.isfile(imgpath)):
        topic_model.visualize_topics().write_image(strs+".png", width=1920, height=1080)
    else:
        os.mkdir(imgpath)
        topic_model.visualize_topics().write_image(strs+".png", width=1920, height=1080)           

def main():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    physical_device = tf.config.experimental.list_physical_devices('GPU')
    print(f'Device found : {physical_device}')
    if not(tf.config.experimental.get_memory_growth(physical_device[0])):
        tf.config.experimental.set_memory_growth(physical_device[0],True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()
    print(torch.cuda.current_device())
#Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    parser = argparse.ArgumentParser(description='Train BertTopics')
    parser.add_argument('--years', default=[["all"]])
    parser.add_argument('--categories',default=[['all']])
    parser.add_argument('--datafile', default='../arxiv_data.json')
    args = parser.parse_args()
    datadf=import_data(args.datafile)
    cleanedDF=clean_text(datadf,'abstract', 'abstract')

    selectedDFs,DFstrings= select_data_by(cleanedDF,args.years,args.categories)
    print('checking select method, ',selectedDFs)
    for trainDF,st in zip(selectedDFs,DFstrings):
        print('BERTOPIC on ',trainDF)
        print(type(trainDF['abstract'].tolist()))
        train_BERTopic(trainDF['abstract'].tolist(),st)
if __name__ == "__main__":
    main()