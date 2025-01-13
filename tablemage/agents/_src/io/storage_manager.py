from llama_index.core.indices import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core import StorageContext, Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.schema import TextNode
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from json import dumps
import os

from ...._src.display.print_utils import suppress_std_output
from ..options import options
from ..llms.utils import describe_image
from .._debug.logger import print_debug
from ...._src.utils.serialize import prepare_for_json


# set the environment variable to disable parallelism for FastEmbedEmbedding
# parallelism conflicts with the Python environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"
with suppress_std_output():
    Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")


io_path = Path(__file__).resolve().parent
img_store_path = io_path / "_img_store"
img_store_path.mkdir(exist_ok=True)

table_store_path = io_path / "_table_store"
table_store_path.mkdir(exist_ok=True)

vector_store_path = io_path / "_vector_store"
vector_store_path.mkdir(exist_ok=True)

obj_vector_store_path = io_path / "_obj_vector_store"
obj_vector_store_path.mkdir(exist_ok=True)

log_path = io_path / "_log"
if log_path.exists():
    for log in log_path.iterdir():
        log.unlink()
else:
    log_path.mkdir(exist_ok=True)
    (log_path / "_log.log").touch()


class ObjectWrappingNode:
    def __init__(self, obj: object, description: str):
        self.obj = obj
        self.description = description

    def __str__(self):
        return self.description + "\n" + str(self.obj)


class StorageManager:
    def __init__(self, multimodal: bool = True, vectorstore: bool = False):
        """Initializes the StorageManager object.

        Parameters
        ----------
        multimodal : bool
            Whether to use a multimodal LLM, by default True.

        vectorstore : bool
            Whether to store data in a vector store, by default False.
        """
        self._llm = options.llm_build_function()
        self._use_vectorstore = vectorstore
        if options.multimodal and multimodal:
            self._multimodal_llm = options.multimodal_llm_build_function()
        else:
            self._multimodal_llm = None
        self.reset()

    def add_obj(self, obj: object, description: str) -> object:
        """Store an object in the object index.

        Parameters
        ----------
        obj : object
            Object to add to the object index.

        description : str
            Description of the object.

        Returns
        -------
        object
            The input object, verbatim.
        """
        if self._use_vectorstore:
            print_debug(f"Adding object to object vector store: {obj}")
            self._obj_index.insert_object(
                obj=ObjectWrappingNode(obj, description),
            )
        return obj

    def retrieve_obj(self, query: str) -> object | None:
        """Retrieve an object from the object index.

        Parameters
        ----------
        query : str
            Query to retrieve an object from the object index.

        Returns
        -------
        object
            The object retrieved from the object index. If no object is found,
            returns None.
        """
        if self._use_vectorstore:
            print_debug(f"Retrieving object from object vector store: {query}")
            retrieved: ObjectWrappingNode = self._obj_index.as_retriever(
                similarity_top_k=1
            ).retrieve(query)[0]
            return retrieved.obj
        return None

    def add_str(self, text: str) -> str:
        """Store text in the vector index.

        Parameters
        ----------
        text : str
            Text to add to the vector index.

        Returns
        -------
        str
            The input text, verbatim.
        """
        if self._use_vectorstore:
            print_debug(f"Adding text to vector store: {text}")
            self._index.insert_nodes(nodes=[TextNode(text=text)])
        return text

    def add_figure(
        self,
        fig: plt.Figure,
        text_description: str,
        augment_text_description: bool = True,
    ) -> tuple[str, Path]:
        """Store a figure as an image in the vector index.

        Parameters
        ----------
        fig : plt.Figure
            Figure to add to the vector index.

        text_description : str
            Description of the figure.

        augment_text_description : bool
            Whether to augment the text description with a vision model,
            by default True

        Returns
        -------
        str
            Description of the figure.

        str
            Path to the image.
        """
        img_path = img_store_path / f"{self._img_counter}.png"

        fig.savefig(img_path, dpi=400)
        self._img_counter += 1

        if augment_text_description and self._multimodal_llm is not None:
            text_description += "\n" + describe_image(
                multimodal_model=self._multimodal_llm,
                image_path=img_path,
                text_description=text_description,
            )
        else:
            text_description += "\n"
            text_description += "A detailed description is unavailable."

        text_description = (
            f"Path to image: {img_path}\n\n" + f"Description: {text_description}"
        )

        if self._use_vectorstore:
            print_debug(f"Adding figure to vector store: {img_path}")
            self._index.insert_nodes(
                nodes=[
                    TextNode(
                        text=text_description,
                        metadata={
                            "path": str(img_path),
                        },
                    )
                ]
            )

        return text_description, img_path

    def add_table(
        self, table: pd.DataFrame, add_to_vectorstore: bool = True
    ) -> tuple[str, Path]:
        """Store a pandas DataFrame in the vector index.
        But also stores pickled DataFrame in disk.

        Parameters
        ----------
        table : pd.DataFrame
            DataFrame to add to the vector index.

        add_to_vectorstore : bool
            Whether to add the DataFrame to the vector store, by default True.
            May want to set to False if a custom dict including the DataFrame
            is to be added to the vector store (e.g. use add_dict instead).

        Returns
        -------
        str
            The input DataFrame, verbatim.

        str
            Path to the pickled DataFrame.
        """
        table_path = table_store_path / f"{self._table_counter}.pkl"
        table.to_pickle(table_path)
        self._table_counter += 1
        str_res = dumps(prepare_for_json(table.to_dict("index")))
        str_res = f"Path to table: {str(table_path)}\n\n" + str_res
        if self._use_vectorstore and add_to_vectorstore:
            print_debug(f"Adding table to vector store: {table_path}")
            self._index.insert_nodes(
                nodes=[
                    TextNode(
                        text=str_res,
                        metadata={
                            "path": str(table_path),
                        },
                    )
                ]
            )
        return str_res, table_path

    def reset(self):
        # delete all images
        for img in img_store_path.iterdir():
            img.unlink()
        self._img_counter = 0

        # delete all tables
        for table in table_store_path.iterdir():
            table.unlink()
        self._table_counter = 0

        for log in log_path.iterdir():
            log.unlink()
        (log_path / "_log.log").touch()

        if self._use_vectorstore:
            _, storage_context = self.setup_vector_store(path=vector_store_path)
            self._index = VectorStoreIndex.from_documents(
                documents=[], storage_context=storage_context
            )
            self._retriever = self._index.as_retriever(similarity_top_k=1)
            self._query_engn = self._index.as_query_engine(llm=self._llm)

            _, obj_storage_context = self.setup_vector_store(path=obj_vector_store_path)

            self._obj_index = ObjectIndex.from_objects(
                objects=[],
                index_cls=VectorStoreIndex,
                storage_context=obj_storage_context,
            )

    @property
    def retriever(self):
        if not self._use_vectorstore:
            raise ValueError("Vector store not in use.")
        return self._retriever

    @property
    def query_engine(self):
        if not self._use_vectorstore:
            raise ValueError("Vector store not in use.")
        return self._query_engn

    def setup_vector_store(
        self, path: Path = vector_store_path
    ) -> tuple[SimpleVectorStore, StorageContext]:
        vector_store = SimpleVectorStore().persist(
            persist_path=path / "vector_store.json"
        )
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
        )
        return vector_store, storage_context
