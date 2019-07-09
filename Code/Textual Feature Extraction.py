import glob
import os.path
import numpy as np
import pandas as pd
from pandas import DataFrame
from collections import OrderedDict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import javalang
from pathlib import Path

class Parser:
    """Class containing parsers"""
    
    __slots__ = ['name']
    
    def __init__(self, name):
        self.name = name

   
    def src_vectorizer(self):
        """Parse source code directory of a program and collect
        its java files.
        """
        
        parse_tree = None
        
        file = open(self.name,'r')
        #print(file.read())
        print('-----------------------------------------------------')
        vectorizer = CountVectorizer()
        # Looping to parse each source file
        
        for x in range (1):
            
            src = file.read()
            print(src)
            
                
            # Placeholder for different parts of a source file
            comments = ''
            class_names = []
            attributes = []
            method_names = []
            variables = []

            # Source parsing
            
            try:
                parse_tree = javalang.parse.parse(src)
                for path, node in parse_tree.filter(javalang.tree.VariableDeclarator):
                    if isinstance(path[-2], javalang.tree.FieldDeclaration):
                        attributes.append(node.name)
                    elif isinstance(path[-2], javalang.tree.VariableDeclaration):
                        variables.append(node.name)
            except:
                pass
         
            # Trimming the source file
            ind = False
            if parse_tree:
                if parse_tree.imports:
                    last_imp_path = parse_tree.imports[-1].path
                    src = src[src.index(last_imp_path) + len(last_imp_path) + 1:]
                elif parse_tree.package:
                    package_name = parse_tree.package.name
                    src = src[src.index(package_name) + len(package_name) + 1:]
                else:  # There is no import and no package declaration
                    ind = True
            # javalang can't parse the source file
            else:
                ind = True
        
        # create the transform
        vectorizer = TfidfVectorizer()
        # tokenize and build vocab
        vectorizer.fit([str(parse_tree)])
        # summarize
        print('---------------------------check 2----------------------------------')
        print(vectorizer.vocabulary_)
        vector = vectorizer.transform([str(parse_tree)])
        print(vector)
        print('---------------------check 3-------------------------------------------------------------')
        a=np.array(vector.toarray())
        print(a)
        print('---------------------check 4-------------------------------------------------------------')
        df = DataFrame(a)
        print(df)
        df.to_csv(r'godclasstextualdata.csv',mode='a',header=False)

        
        

if __name__ == '__main__':
    parser = Parser('')

    srcfilesnames = pd.read_csv('allnames.csv')
    names= srcfilesnames.iloc[0:len(srcfilesnames),0]
    names=np.array(names)
    print(names)
    foundsrc=0
    notdound=0
    for i in range(names.shape[0]):
        try:
            fh = open(names[i]+".java", 'r')
            foundsrc=foundsrc+1
            parser = Parser(names[i]+'.java')
            parser.src_vectorizer()
            print("found")
        except FileNotFoundError:
            notdound=notdound+1
            missingarray=(['notfound'])
            df = DataFrame(missingarray)
            df.to_csv(r'godclasstextualdata.csv',mode='a',header=False)
            print("notfound")


    print("f ",foundsrc)
    print("notfound ",notdound) 
    print(names)
    


    
    

    
