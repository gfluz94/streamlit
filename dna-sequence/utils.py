from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


class DNA():

    def __init__(self, sequence):
        self.__sequence = self.__preprocess_sequence(sequence)
        self.__nucleotides = {
            "A": "Adenine",
            "T": "Thymine",
            "G": "Guanine",
            "C": "Cytosine"
        }

    @property
    def sequence(self):
        return self.__sequence

    def __preprocess_sequence(self, sequence):
        return "".join(sequence.split("\n")[1:])

    def __map_to_rna(self, nucleotide: str) -> str:
        key2value = {
            "A": "A",
            "T": "U",
            "G": "G",
            "C": "C"
        }
        return key2value[nucleotide]

    def count_nucleotides(self) -> dict:
        return {key: self.sequence.count(key) for key in self.__nucleotides.keys()}

    def get_dataframe(self) -> pd.DataFrame:
        dict_count = self.count_nucleotides()
        df = pd.DataFrame({
            "Nucleotide": dict_count.keys(),
            "Count": dict_count.values()
        })
        return df.sort_values(by="Count", ascending=False).set_index("Nucleotide")

    def explanatory_text(self) -> List[str]:
        return [
            f"There are {count} {self.__nucleotides[n]} ({n})" for n, count in self.count_nucleotides().items()
        ]

    def get_mrna(self):
        return "".join([self.__map_to_rna(n) for n in self.sequence])


def read_image(image_path: str) -> Image:
    return Image.open(image_path)

def barplot_seaborn(dna: DNA):
    fig = plt.figure(figsize=(12, 6))
    df = dna.get_dataframe()
    sns.barplot(y=df.Count, x=df.index)
    return fig
