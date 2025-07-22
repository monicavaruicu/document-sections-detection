from categoryScraper import scrape_categories
from paperURLsScraper import scrape_papers_urls
from paperContentScraper import scrape_papers

subject = 'Computer Science'
categories_file = 'arxiv_categories.csv'
papers_file = 'arxiv_papers.csv'

def main():

    scrape_categories(subject, categories_file)
    scrape_papers_urls(categories_file, papers_file)
    scrape_papers(papers_file)

if __name__ == '__main__':
    main()
