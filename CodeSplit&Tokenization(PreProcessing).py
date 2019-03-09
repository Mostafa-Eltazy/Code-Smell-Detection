import glob
import os.path
from collections import OrderedDict

#import xmltodict
import javalang
import pygments
from pygments.lexers import JavaLexer
from pygments.token import Token

class SourceFile:
    """Class representing each source file"""
    
    __slots__ = ['all_content', 'comments', 'class_names', 'attributes',
                 'method_names', 'variables', 'file_name', 'pos_tagged_comments',
                 'exact_file_name', 'package_name']
    
    def __init__(self, all_content, comments, class_names, attributes,
                 method_names, variables, file_name, package_name):
        self.all_content = all_content
        self.comments = comments
        self.class_names = class_names
        self.attributes = attributes
        self.method_names = method_names
        self.variables = variables
        self.file_name = file_name
        self.exact_file_name = file_name[0]
        self.package_name = package_name
        self.pos_tagged_comments = None


class Parser:
    """Class containing different parsers"""
    
    __slots__ = ['name', 'src']
    
    def __init__(self, name,src):
        self.name = name
        self.src = src
        
        


    def src_parser(self):
        """Parse source code directory of a program and collect
        its java files.
        """
        
        # Getting the list of source files recursively from the source directory
        src_addresses = glob.glob(str(self.src) + '/**/*.java', recursive=True)
        # Creating a java lexer instance for pygments.lex() method
        java_lexer = JavaLexer()
        
        src_files = OrderedDict()
        
        # Looping to parse each source file
        allLexedFiles = []
        for src_file in src_addresses:
            with open(src_file) as file:
                src = file.read()
                
            # Placeholder for different parts of a source file
            comments = ''
            class_names = []
            attributes = []
            method_names = []
            variables = []

            # Source parsing
            parse_tree = None
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
            
            # Lexically tokenize the source file
            lexed_src = pygments.lex(src, java_lexer)
            allLexedFiles.append(lexed_src)
            #print(lexed_src)
            for i, token in enumerate(lexed_src):
                if token[0] in Token.Comment:
                    if ind and i == 0 and token[0] is Token.Comment.Multiline:
                        src = src[src.index(token[1]) + len(token[1]):]
                        continue
                    comments += token[1]
                elif token[0] is Token.Name.Class:
                    class_names.append(token[1])
                    
                    
                elif token[0] is Token.Name.Function:
                    method_names.append(token[1])
                    
            

        
        
        return allLexedFiles




if __name__ == '__main__':
    parser = Parser('my project', 'D:\Bue\EL senior fucking year\Semester one\ADV.OO.SE\Project\AOOSE-master')
    d = parser.src_parser()
    