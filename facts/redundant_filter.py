from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Chroma 
from langchain.schema import BaseRetriever

# here we are create a retriever filter that will filter the similar documents. 
class RedundantFilterRetriever(BaseRetriever): 
    # create a internal variable for embeddings that will be set from outside calling fucntion
    embeddings: Embeddings
    vector_db: Chroma  # fulling set up database with persistent directory. 

    def get_relevant_documents(self, query):
        # do caluclation of embeddings 
        emb = self.embeddings.embed_query(query) 

        # take the embeddings and feed it into max_marginal_relevant_search_by_vector
        results = self.vector_db.max_marginal_relevance_search_by_vector(
            embedding = emb, # use the already claculated embedding earlier to 
                             # find and filter similar documents 
            lambda_mult = 0.8  # tolerance for similarity 
        )
        return results
    
    async def aget_relevant_documents(self): 
        return []