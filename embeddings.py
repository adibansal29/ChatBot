energysage_lst_files=['com-solar-.txt', 'com-community-solar-.txt', 'com-solar-best-solar-panels-complete-ranking-.txt', 'com-local-data-solar-panel-cost-.txt', 'com-energy-storage-.txt', 'com-energy-storage-best-home-batteries-.txt', 'com-energy-storage-battery-backup-power-vs-generators-which-is-right-for-you-.txt', 'com-heat-pumps-.txt', 'com-heat-pumps-costs-and-benefits-air-source-heat-pumps-.txt', 'com-heat-pumps-how-do-heat-pumps-work-.txt', 'com-ev-charging-.txt', 'com-electric-vehicles-.txt', 'com-ev-charging-ev-charging-stations-.txt', 'com-ev-charging-electric-vehicle-charging-cost-.txt', 'com-electricity-.txt', 'com-energy-management-.txt', 'com-energy-efficiency-.txt', 'com-energy-products-.txt', 'com-business-solutions-commercial-solar-benefits-.txt', 'com-business-solutions-solar-nonprofit-benefits-financing-.txt', 'com-business-solutions-energy-storage-for-businesses-.txt', 'com-solar-how-to-pay-for-solar-.txt', 'com-solar-solar-loans-.txt', 'com-solar-solar-leases-.txt', 'com-shop-clean-energy-savings-and-tax-incentives-.txt', 'com-blog-.txt', 'com-blog-.txt', 'com-energy-storage-tesla-powerwall-how-much-home-can-you-run-on-it-for-how-long-.txt', 'com-solar-solar-panel-efficiency-cost-over-time-.txt', 'com-blog-.txt', 'com-solar-are-solar-panels-worth-it-.txt', 'com-shop-home-solar-.txt', 'com-energy-advisors-.txt', 'com-partners-corporations.txt', 'com-your-marketplace-.txt', 'com-shop-home-solar-.txt', 'com-shop-home-solar-.txt', 'com-shop-home-solar-.txt', 'com-shop-home-solar-.txt', 'com-solar-.txt', 'com-local-data-solar-rebates-incentives-.txt', 'com-energy-storage-.txt', 'com-shop-home-solar-.txt', 'com-shop-community-solar-.txt', 'com-shop-community-solar-.txt', 'com-community-solar-.txt', 'com-shop-community-solar-.txt', 'com-local-data-solar-panel-cost-.txt', 'com-shop-community-solar-.txt', 'com-shop-community-solar-.txt', 'com-community-solar-.txt', 'com-shop-heat-pumps-.txt', 'com-shop-heat-pumps-.txt', 'com-heat-pumps-.txt', 'com-heat-pumps-heat-pump-incentives-.txt', 'com-shop-heat-pumps-.txt', 'com-energy-storage-.txt', 'com-energy-storage-.txt', 'com-energy-storage-.txt', 'com-energy-storage-benefits-of-storage-energy-storage-incentives-.txt', 'com-ev-charging-.txt', 'com-ev-charging-.txt', 'com-ev-charging-.txt', 'com-electric-vehicles-.txt', 'com-business-solutions-commercial-solar-benefits-.txt', 'com-business-solutions-solar-nonprofit-benefits-financing-.txt', 'com-business-solutions-energy-storage-for-businesses-.txt', 'com-your-marketplace-.txt', 'com-your-marketplace-.txt', 'com-shop-home-solar-.txt', 'com-shop-community-solar-.txt', 'com-shop-heat-pumps-.txt', 'com-energy-storage-.txt', 'com-shop-clean-energy-savings-and-tax-incentives-.txt', 'com-local-data-solar-companies-ca-.txt', 'com-local-data-solar-companies-ca-.txt', 'com-solar-.txt', 'com-heat-pumps-.txt', 'pumps.energysage.com-.txt', 'com-community-solar-.txt', 'com-solar-.txt', 'com-heat-pumps-.txt', 'pumps.energysage.com-.txt', 'com-community-solar-.txt', 'com-shop-home-solar-.txt', 'com-shop-community-solar-.txt', 'com-shop-heat-pumps-.txt', 'com-energy-storage-.txt', 'com-ev-charging-.txt', 'com-other-clean-options-.txt', 'com-blog-.txt', 'com-shop-home-solar-.txt']
solarreview_lst_files=['-solarreviews.com-going-solar-with-your-utility.txt', '-solarreviews.com-solar-companies.txt', '-solarreviews.com-solar-panel-reviews#expert-review.txt', '-solarreviews.com-solar-battery-reviews.txt', '-solarreviews.com-solar-inverter-reviews.txt', '-solarreviews.com-solar-panel-cost.txt', '-solarreviews.com-blog-how-to-calculate-your-solar-payback-period.txt', '-solarreviews.com-blog-what-are-the-best-solar-panels-to-buy-for-your-home.txt', '-solarreviews.com-press-solarreviews-nabcep-announce-renewed-partnership.txt', '-solarreviews.com-press-solarreviews-releases-top-10-states-for-home-solar.txt', '-solarreviews.com-press-solarreviews-releases-solar-industry-survey-results.txt', '-solarreviews.com-press-solarreviews-launches-2022-solar-industry-survey.txt', '-solarreviews.com-press-solarreviews-releases-best-solar-panel-manufacturer-list-2023.txt', '-solarreviews.com-press-solarreviews-and-sei-announce-renewed-partnership.txt', '-solarreviews.com-press-solarreviews-and-nabcep-renew-partnership.txt', '-solarreviews.com-press-solarreviews-releases-best-solar-panel-brands-list-2024.txt', '-solarreviews.com-press-solarreviews-ira-bill-survey-results.txt', '-solarreviews.com-press-solarreviews-acquires-fixr.txt', '-solarreviews.com-press-solarreviews-releases-senate-bill-1024-survey-results.txt', '-solarreviews.com-press-solarreviews-survey-cpuc-proposal.txt', '-solarreviews.com-press-solarreviews-releases-solar-battery-adoption-report.txt', '-solarreviews.com-press-solarreviews-releases-best-solar-manufacturers-ranking.txt', '-solarreviews.com-press-solarreviews-releases-new-solar-calculator.txt', '-solarreviews.com-press-solarreviews-releases-ev-consumer-guide.txt', '-solarreviews.com-press-solarreviews-and-nabcep-exclusive-partnership.txt', '-solarreviews.com-press-solarreviews-and-beacn-release-largest-home-solar-survey.txt', '-solarreviews.com-press-solarreviews-announces-new-employee-incentive-to-promote-sustainability.txt', '-solarreviews.com-press-solar-energy-international-solarreviews-partnership.txt', '-solarreviews.com-press-solarreviews-launches-2024-solar-industry-survey.txt', 'solarreviews.com-press-solarreviews-cautions-consumers-to-understand-tesla-solar-risks.txt']
energysage_lst_files.extend(solarreview_lst_files)

import os
import cohere
from dotenv import load_dotenv, find_dotenv

# Load the API key from the .env file
_ = load_dotenv(find_dotenv())
cohere.api_key  = os.environ['COHERE_API_KEY']

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Chroma

# Load the documents
asd=set(energysage_lst_files)
docs=[]
for url_file in asd:
    loader = TextLoader(url_file)
    docs.extend(loader.load())

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
# Split the documents into chunks
splits=text_splitter.split_documents(docs)

# Create the vector database
persist_directory = 'Enphase_DB'

embedding=CohereEmbeddings()
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

print(vectordb._collection.count())


