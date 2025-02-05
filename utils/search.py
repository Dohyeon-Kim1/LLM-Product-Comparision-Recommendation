import re
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.tools import DuckDuckGoSearchResults


SEARCH = DuckDuckGoSearchResults(max_results=3, output_format="list")


def crawling(response, max_length = 20000):
    try:
        links = [result['link'] for result in response
                 if 'Error' not in result['title']
                 and 'youtube' not in result['link'].lower()]

        if not links:
            return ""

        loader = AsyncHtmlLoader(links)
        docs = loader.load()
        bs_transformer = BeautifulSoupTransformer()
        docs_transformed = bs_transformer.transform_documents(
            docs,
            unwanted_tags=[
                "script", "style", "nav", "footer", "header",
                "aside", "button", "form", "input", "iframe",
                "noscript", "img", "svg", "path"
            ],
            tags_to_extract=["div.content","main.content","p"],
            unwanted_classnames=[
                "advertisement", "sidebar", "menu", "navigation",
                "footer", "header", "social", "comments"
            ],
            remove_lines=True,
            remove_comments=True
        )

        cleaned_texts = []
        for doc in docs_transformed:
            text = doc.page_content
            text = re.sub(r'\(https?://[^)]+\)', "", text)
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()

        if len(text) > 100:
            cleaned_texts.append(text)

        combined_text = ' '.join(cleaned_texts)

        if len(combined_text) > max_length:
            final_text = combined_text[:max_length].rsplit('.', 1)[0] + '.'
        else:
            final_text = combined_text

        return final_text

    except Exception as e:
        print(f"Error in crawling: {str(e)}")
        return ""


def webSearch(query):
  try:
    response = SEARCH.invoke(query)
    context = crawling(response)
    return context

  except Exception as e:
    print(f"Error processing query: {str(e)}")
    return ""