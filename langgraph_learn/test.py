from pymed_paperscraper import PubMed

pubmed = PubMed(tool="AlifScraper", email="aliffaturrohman11@gmail.com")
results = pubmed.query("tubercolosis", max_results=1)

for i, article in enumerate(results, start=1):
    print(f"=== Artikel {i} ===")
    print("Judul   :", article.title)
    print("Jurnal  :", article.journal)
    print("DOI     :", article.doi)
    print("Keyword :", article.keywords)
    print("Abstract :", article.abstract)
