import base64
from io import BytesIO
import os
from tkinter.filedialog import asksaveasfile, asksaveasfilename
import webbrowser
import pandas as pd
from rdkit.Chem import Draw
from rdkit.Chem.rdCoordGen import AddCoords
from rdkit import Chem
import shutil
import time
from time import sleep
from bs4 import BeautifulSoup as bs
from copy import copy
from tqdm import tqdm

class Mostrador():
    def __init__(self) -> None:
        self.data = None
        self.tempdir = os.path.join("./src/mostrador/temp/", str(int(time.time())))
        os.makedirs(self.tempdir)

        self.filesdir = os.path.join(self.tempdir, "index_files")


    def add(self, data: pd.DataFrame, smiles="Smiles", title="Smiles", subtitle=None):
        aux = data.copy(deep=True)
        # checkar se tem as colunas necessárias
        if not smiles or smiles not in data:
            print("Erro a adicionar ao mostrador: Falta a coluna", smiles)
            return
        if title:
            if title not in data:
                print("Erro a adicionar ao mostrador: Falta a coluna", title)
                return
            aux["_titulo"] = data[title].map(str)
        if subtitle:
             #usar este sep para ter a certeza q nao ha conflitos
            if type(subtitle) == str:
                subtitle = [subtitle]
            
            aux["_subtitulo"] = subtitle[0] + ":--->:" + data[subtitle[0]].map(str)
            for sub in subtitle[1:]:
                if sub not in data:
                    print("Erro a adicionar ao mostrador: Falta a coluna", sub)
                    return
                aux["_subtitulo"] += "|||" + sub + ":--->:" + data[sub].map(str)
            

        aux["Smiles"] = data[smiles] #normalizar os datasets
        aux["_hash"] = data[smiles].apply(hash) #------ ATENÇÃO ------- a func hash() nao mantem os mesmos valores entre execuções de python, é randomizada cada vez que o interpretador é inicializado, mas como aqui so esta a aser usada para substituir os carateres estranhos dos smiles por nums nao ha problema
        if self.data is None:
            self.data = aux
        else:
            self.data = self.data.append(aux, ignore_index=True)


    def clean(self):
        self.data = None

    def render(self, sort_by=None):
        if sort_by:
            self.data = self.data.sort_values(sort_by)
        print("A renderizar index.html...")
        print("A gerar imagens...")

        #preencher html    
        with open("./src/mostrador/index.html", "r") as template_file:
            template = bs(template_file, "html.parser")
        
        with open("./src/mostrador/cartao_molecula_elemento.html", "r") as elemento_molecula:
            e_molecula = bs(elemento_molecula, "html.parser")

        div_moleculas = template.find(id="div-moleculas")
        print("A popular index.html...")
        template.find(id = "n_compostos").string = str(self.data.shape[0]) + " Compounds"

        #COISAS Q SE FAZEM P CADA MOLECULA
        for i, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            cartao_molecula = copy(e_molecula)
            main_div = cartao_molecula.find()
            main_div["id"] = i
            main_div["data-details"] = row.to_json()
            #cartao_molecula.find(id="img-molecula")['src'] = os.path.join("imagens", str(row["_hash"]) + ".png") # ATENÇÃOOOOO, QUANDO O INDEX.HTML CORRE NO BROWSER JA NAO ESTAS COM O MESMO CWD, PELO Q O ./ REMETE PARA A PASTA TEMP/QQCOISA---------------------------------------------OLHAAA
            encoded_string = Mostrador.__render_image(row["Smiles"])
            cartao_molecula.find(id="img-molecula")['src'] = "data:image/png;base64," + encoded_string
            cartao_molecula.find(id="label-id-molecula").string = str(i + 1)
            tag0, tag1 = cartao_molecula.find_all(id="BCK-cards-ID_MOLECULA-select")
            tag0["id"] = "BCK-cards-" + str(i) + "-select"
            tag1["for"] = "BCK-cards-" + str(i) + "-select"


            if pd.notna(row["_titulo"]):
                cartao_molecula.find(id="titulo").string = row["_titulo"]
            if pd.notna(row["_subtitulo"]):
                parent_subtitulo_elem = cartao_molecula.find(id="subtitulos")
                for s in row["_subtitulo"].split("|||"):
                    subtitulo_elem = cartao_molecula.new_tag("p")
                    t_subt, str_subt = s.split(":--->:")
                    t_subt_tag = cartao_molecula.new_tag("b")
                    t_subt_tag.string = t_subt + ": "
                    subtitulo_elem["class"] = "p-oneline"
                    subtitulo_elem.string = str_subt
                    subtitulo_elem.insert(0, t_subt_tag)
                    parent_subtitulo_elem.append(subtitulo_elem)
            div_moleculas.append(cartao_molecula)
    
        with open(os.path.join(self.tempdir, "index.html"), "w") as out:
            out.write(str(template))



    def show(self, sort_by=None):
        index_file = os.path.join(self.tempdir, "index.html")
        if not os.path.exists(index_file): #redundancia para o caso de me esquecer
            print("Não existia, a renderizar...")
            self.render(sort_by)
        webbrowser.open(index_file)
        sleep(2) #como isto abre threads q n controlo o melhor é usar um sleep para dar tempo

    def __render_image(smile):
        mol = Chem.MolFromSmiles(smile)
        AddCoords(mol)
        img = Draw.MolToImage(mol=mol)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())
        return img_str.decode("UTF-8")

    def save(self, path=None):
        if path == None:
            path = asksaveasfilename(initialfile="resultados" + time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime()) + ".html", initialdir= "./",filetypes=[('HTML File', '*.html'), ('All Files', '*.*')])
        if path:
            shutil.copyfile(os.path.join(self.tempdir, "index.html"), path)

    def __del__(self):
        shutil.rmtree("./src/mostrador/temp/", ignore_errors=True)  # descomentar para apagar a pasta temporaria a cada execucao e poupar espaço





if __name__ == '__main__':
    disp = Mostrador()
    data = pd.read_csv("./raw_data/test_dataset.csv", sep=";")
    disp.add(data, subtitle="AlogP")
    disp.add(data, subtitle=["AlogP", "Molecular Weight", "Aromatic Rings"])
    # disp.add(data)
    # disp.add(data, title="AlogP", subtitle="Smiles")
    # print(disp.data[["_subtitulo", "_titulo"]])
    # print(data)
    # print(disp.data)
    disp.render()#sort_by="AlogP")
    disp.show()
    # disp.save("./resultados.html")
    # sleep(10)