# HUMAN-COMPUTER INTERACTION 2025-2026

## Descrizione  
L'obiettivo del progetto è analizzare e visualizzare un ampio set di dati relativi agli anime tramite l'uso di Python, Pandas, Seaborn e librerie varie.  

## Come eseguire il progetto  
Per caricare automaticamente le librerie utilizzate possiamo seguire due metodi tramite terminale, all'interno della cartella principale del progetto:

### **Tramite requirements.txt** (due scelte)  
**A. Installazione diretta**  
* ```pip install -r requirements.txt```   

**B. Per mantenere il sistema pulito ed evitare conflitti tra librerie, si può creare un ambiente isolato:**  
* Crea ambiente: ```python -m venv venv```  
* Attiva: Windows: ```venv\Scripts\activate```  Mac/Linux: ```source venv/bin/activate```   
* Installa: ```pip install -r requirements.txt```  

### **Tramite conda**:  
Se utilizzi Anaconda o Miniconda, puoi rigenerare l'ambiente esatto tramite il file YAML:  
* ```conda env create -f environment.yml```  
* ```conda activate hci_project```  

Il Jupyter Notebook sarà così pronto per essere eseguito e caricherà automaticamente tutte le librerie necessarie.
