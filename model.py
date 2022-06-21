# =============================================================================
# Main file  
# =============================================================================

from src import dataloader,model

def run():
    model.load_data() # twitter-uni, imdb, arxiv-10
    """Builds model, loads data, trains and evaluates"""
    model.build()
    model.train()
    model.evaluate()

if __name__ == '__main__':
    run()