import os
import pandas as pd
import itertools
import random
import traceback
import optuna  
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, send_from_directory, flash, redirect, url_for
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, CopulaGANSynthesizer, GaussianCopulaSynthesizer
from sdv.metadata.single_table import SingleTableMetadata
from sdmetrics.reports.single_table import QualityReport


# CLASSE 

class DataSynthesizer:
    # ... (Copiez-collez votre classe DataSynthesizer ici, elle n'a pas besoin de changer)
    # ... Pour la concision, je ne la répète pas, mais elle doit être présente.
    """
    Une classe pour gérer tout le processus de synthèse de données :
    - Chargement des données et métadonnées
    - Configuration et entraînement des modèles
    - Génération de données synthétiques
    - Sauvegarde des données de sortie
    """

    def __init__(self, upload_folder, download_folder):
        self.upload_folder = upload_folder
        self.download_folder = download_folder
        self.model = None
        self.original_data = None
        self.synthetic_data = None
        self.combined_data = None
        self.metadata = None
        self.quality_report_object = None
        self.detailed_metrics = {}

    def load_data(self, file_storage):
        """Charge le fichier CSV et détecte les métadonnées."""
        if not file_storage or not file_storage.filename.endswith('.csv'):
            raise ValueError("Fichier non valide. Veuillez fournir un fichier CSV.")

        filename = secure_filename(file_storage.filename)
        self.original_filepath = os.path.join(self.upload_folder, filename)
        file_storage.save(self.original_filepath)

        self.original_data = pd.read_csv(self.original_filepath)
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(self.original_data)
        
        print("Données chargées et métadonnées détectées avec succès.")
        return filename

    def train_model(self, model_params):
        """Configure et entraîne le modèle sélectionné."""
        model_type = model_params.get('model_type', 'ctgan')

        # Créer une copie pour éviter de modifier le dictionnaire original
        params_copy = model_params.copy()

        if model_type == 'ctgan':
            self.model = self._create_ctgan_model(params_copy)
        elif model_type == 'tvae':
            self.model = self._create_tvae_model(params_copy)
        elif model_type == 'copulagan':
            self.model = self._create_copulagan_model(params_copy)
        elif model_type == 'gaussiancopula':
            self.model = self._create_gaussiancopula_model(params_copy)
        else:
            raise ValueError(f"Type de modèle non supporté : {model_type}")

        print(f"Début de l'entraînement du modèle {model_type.upper()}...")
        self.model.fit(self.original_data)
        print("Entraînement terminé.")

    def _create_ctgan_model(self, params):
        return CTGANSynthesizer(metadata=self.metadata, enforce_min_max_values=True,
                                epochs=int(params.get('epochs', 300)),
                                batch_size=int(params.get('batch_size', 500)),
                                generator_lr=float(params.get('generator_lr', 2e-4)),
                                discriminator_lr=float(params.get('discriminator_lr', 2e-4)),
                                embedding_dim=int(params.get('embedding_dim', 128)),
                                generator_dim=params.get('generator_dim', (256, 256)),
                                discriminator_dim=params.get('discriminator_dim', (256, 256)),
                                verbose=False)

    def _create_tvae_model(self, params):
        return TVAESynthesizer(metadata=self.metadata, enforce_min_max_values=True,
                               epochs=int(params.get('epochs', 300)),
                               batch_size=int(params.get('batch_size', 500)),
                               embedding_dim=int(params.get('embedding_dim', 128)),
                               compress_dims=params.get('compress_dims', (128, 128)),
                               decompress_dims=params.get('decompress_dims', (128, 128)),
                               l2scale=float(params.get('l2scale', 1e-5)),
                               loss_factor=float(params.get('loss_factor', 2)),
                               verbose=False)
    
    def _create_copulagan_model(self, params):
        return CopulaGANSynthesizer(metadata=self.metadata, enforce_min_max_values=True,
                                    epochs=int(params.get('epochs', 300)),
                                    batch_size=int(params.get('batch_size', 500)),
                                    embedding_dim=int(params.get('embedding_dim', 128)),
                                    generator_lr=float(params.get('generator_lr', 2e-4)),
                                    discriminator_lr=float(params.get('discriminator_lr', 2e-4)),
                                    generator_dim=params.get('generator_dim', (256, 256)),
                                    discriminator_dim=params.get('discriminator_dim', (256, 256)),
                                    discriminator_steps=int(params.get('discriminator_steps', 1)),
                                    verbose=False)
    
    def _create_gaussiancopula_model(self, params):
        return GaussianCopulaSynthesizer(metadata=self.metadata, enforce_min_max_values=True,
                                         default_distribution=params.get('default_distribution', 'norm'),
                                         enforce_rounding=(params.get('enforce_rounding') == 'on'))

    def generate_synthetic_data(self, num_rows):
        if not self.model:
            raise RuntimeError("Le modèle doit être entraîné avant de générer des données.")
        self.synthetic_data = self.model.sample(num_rows=int(num_rows))

    def evaluate_generation(self):
        if self.original_data is None or self.synthetic_data is None:
            raise RuntimeError("Les données originales et synthétiques sont nécessaires pour l'évaluation.")
        self.quality_report_object = QualityReport()
        self.quality_report_object.generate(self.original_data, self.synthetic_data, self.metadata.to_dict())

    def save_results(self, original_filename):
        if self.original_data is None or self.synthetic_data is None:
            raise RuntimeError("Les données originales et synthétiques doivent exister pour sauvegarder les résultats.")
        self.combined_data = pd.concat([self.original_data, self.synthetic_data], ignore_index=True)
        base_name, _ = os.path.splitext(original_filename)
        output_filename = f"{base_name}_augmente.csv"
        output_filepath = os.path.join(self.download_folder, output_filename)
        self.combined_data.to_csv(output_filepath, index=False)
        return output_filename

    def get_results_summary(self):
        """Prépare un dictionnaire de résultats à partir de l'objet QualityReport."""
        if self.quality_report_object is None:
            self.evaluate_generation()

        overall_score = self.quality_report_object.get_score()
        column_shape_details = self.quality_report_object.get_details('Column Shapes')
        column_pair_details = self.quality_report_object.get_details('Column Pair Trends')
        detailed_reports_html = {}
        
        if not column_shape_details.empty:
            df = column_shape_details[['Column', 'Metric', 'Score']].copy()
            df['Score'] = df['Score'].apply(lambda x: f"{x:.2%}" if pd.notna(x) else 'N/A')
            detailed_reports_html['Similarité des Distributions (par colonne)'] = df.to_html(classes='table table-sm table-striped', index=False)

        if not column_pair_details.empty:
            df = column_pair_details[['Column 1', 'Column 2', 'Metric', 'Score']].copy()
            df['Score'] = df['Score'].apply(lambda x: f"{x:.2%}" if pd.notna(x) else 'N/A')
            detailed_reports_html['Similarité des Corrélations (par paire)'] = df.to_html(classes='table table-sm table-striped', index=False)
        
        return {'original_rows': len(self.original_data),
                'synthetic_rows': len(self.synthetic_data),
                'total_rows': len(self.combined_data),
                'table_preview': self.synthetic_data.head().to_html(classes='table table-striped', justify='center'),
                'overall_quality_score': f"{overall_score:.2%}" if pd.notna(overall_score) else 'N/A',
                'detailed_reports': detailed_reports_html}


# PARTIE FLASK

# --- Configuration de l'application ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'une-cle-secrete-tres-importante'

UPLOAD_FOLDER = 'uploads'
DOWNLOAD_FOLDER = 'downloads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

data_processor = DataSynthesizer(upload_folder=UPLOAD_FOLDER, download_folder=DOWNLOAD_FOLDER)

# Fonctions pour la Recherche d'Hyperparamètres 

def parse_hyperparams(form_params):
    """Parse les valeurs du formulaire en listes de valeurs, avec des défauts robustes."""
    # Plages par défaut
    defaults = {
        "epochs": [100, 300, 500],
        "batch_size": [250, 500],
        "embedding_dim": [128, 256],
        "generator_lr": [1e-4, 2e-4, 5e-4],
        "discriminator_lr": [1e-4, 2e-4, 5e-4],
        "generator_dim": [(128, 128), (256, 256)],
        "discriminator_dim": [(128, 128), (256, 256)],
        "compress_dims": [(128, 128), (256, 128)],
        "decompress_dims": [(128, 128), (128, 256)],
        "l2scale": [1e-6, 1e-5],
        "loss_factor": [1.0, 2.0, 5.0],
        "discriminator_steps": [1, 5],
        "default_distribution": ["norm", "beta", "gamma"],
    }
    parsed = {}
    for key, default_value in defaults.items():
        form_value = form_params.get(key, "").strip()
        if not form_value:
            parsed[key] = default_value
            continue
        try:
            if key in ["generator_dim", "discriminator_dim", "compress_dims", "decompress_dims"]:
                parsed[key] = [tuple(map(int, x.strip().split(","))) for x in form_value.split(";") if x.strip()]
            elif key in ["generator_lr", "discriminator_lr", "l2scale", "loss_factor"]:
                parsed[key] = [float(x.strip()) for x in form_value.split(",") if x.strip()]
            elif key in ["epochs", "batch_size", "embedding_dim", "discriminator_steps"]:
                parsed[key] = [int(x.strip()) for x in form_value.split(",") if x.strip()]
            elif key == "default_distribution":
                parsed[key] = [x.strip() for x in form_value.split(",") if x.strip()]
            if not parsed.get(key): parsed[key] = default_value
        except (ValueError, TypeError):
            print(f"Erreur de parsing pour '{key}', utilisation de la valeur par défaut.")
            parsed[key] = default_value
    return parsed

def get_relevant_params_for_model(model_type):
    common_params = ['epochs', 'batch_size', 'embedding_dim']
    gan_params = ['generator_lr', 'discriminator_lr', 'generator_dim', 'discriminator_dim']
    if model_type == 'ctgan': return common_params + gan_params
    if model_type == 'tvae': return common_params + ['compress_dims', 'decompress_dims', 'l2scale', 'loss_factor']
    if model_type == 'copulagan': return common_params + gan_params + ['discriminator_steps']
    if model_type == 'gaussiancopula': return ['default_distribution']
    return []

def grid_search(data_processor, model_type, param_grid, num_rows):
    keys, best_score, best_params = list(param_grid.keys()), -1, None
    all_combinations = list(itertools.product(*(param_grid[k] for k in keys)))
    print(f"Grid Search: Test de {len(all_combinations)} combinaisons pour le modèle {model_type}.")
    for i, values in enumerate(all_combinations):
        params = dict(zip(keys, values))
        params['model_type'] = model_type
        print(f"Test [{i+1}/{len(all_combinations)}]: {params}")
        try:
            data_processor.train_model(params)
            data_processor.generate_synthetic_data(num_rows)
            data_processor.evaluate_generation()
            score = data_processor.quality_report_object.get_score()
            if score is not None and score > best_score:
                best_score, best_params = score, params.copy()
        except Exception as e: print(f"Erreur avec params {params}: {e}")
    return best_params, best_score

def random_search(data_processor, model_type, param_grid, num_rows, n_iter=10):
    keys, best_score, best_params = list(param_grid.keys()), -1, None
    print(f"Random Search: Test de {n_iter} combinaisons pour le modèle {model_type}.")
    for i in range(n_iter):
        params = {k: random.choice(param_grid[k]) for k in keys}
        params['model_type'] = model_type
        print(f"Test [{i+1}/{n_iter}]: {params}")
        try:
            data_processor.train_model(params)
            data_processor.generate_synthetic_data(num_rows)
            data_processor.evaluate_generation()
            score = data_processor.quality_report_object.get_score()
            if score is not None and score > best_score:
                best_score, best_params = score, params.copy()
        except Exception as e: print(f"Erreur avec params {params}: {e}")
    return best_params, best_score


def bayesian_search(data_processor, model_type, param_grid, num_rows, n_trials=20):
    """Effectue une recherche bayésienne avec Optuna."""
    print(f"Bayesian Search: Test de {n_trials} essais pour le modèle {model_type}.")
    
    def objective(trial):
        params = {}
        for k, v in param_grid.items():
            params[k] = trial.suggest_categorical(k, v)
        
        params['model_type'] = model_type
        
        try:
            data_processor.train_model(params)
            data_processor.generate_synthetic_data(num_rows)
            data_processor.evaluate_generation()
            score = data_processor.quality_report_object.get_score()
         
            return score if pd.notna(score) else 0.0
        except Exception as e:
            print(f"Erreur avec params {params}: {e}")
            return 0.0  # Un score très bas si l'essai échoue

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

   
    best_params_found = study.best_params
    best_params_found['model_type'] = model_type
    best_score = study.best_value
    
    return best_params_found, best_score

def find_best_hyperparams(data_processor, form_params, num_rows):
    model_type = form_params.get('model_type', 'ctgan')
    search_method = form_params.get('search_method', 'grid')
    
    full_param_grid = parse_hyperparams(form_params)
    relevant_keys = get_relevant_params_for_model(model_type)
    filtered_param_grid = {key: full_param_grid[key] for key in relevant_keys if key in full_param_grid}

    if not filtered_param_grid:
        raise ValueError(f"Aucun hyperparamètre pertinent trouvé ou défini pour le modèle {model_type}.")
    
    print(f"\nLancement de la recherche ({search_method}) pour le modèle '{model_type}'...")
    print(f"Espace de recherche utilisé: {filtered_param_grid}")

    if search_method == 'grid':
        return grid_search(data_processor, model_type, filtered_param_grid, num_rows)

    elif search_method == 'bayesian':
        n_trials = int(form_params.get('random_search_iterations', 20)) # On réutilise le même champ
        return bayesian_search(data_processor, model_type, filtered_param_grid, num_rows, n_trials=n_trials)
    else: 
        n_iter = int(form_params.get('random_search_iterations', 10))
        return random_search(data_processor, model_type, filtered_param_grid, num_rows, n_iter=n_iter)


# --- Routes Flask ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            file_storage = request.files.get('file')
            if not file_storage or file_storage.filename == '':
                flash("Veuillez sélectionner un fichier.")
                return redirect(request.url)
            
            original_filename = data_processor.load_data(file_storage)
            form_params = request.form.to_dict(flat=True)
            num_rows = int(form_params.get('num_rows', len(data_processor.original_data)))

            best_params, best_score = find_best_hyperparams(data_processor, form_params, num_rows)
            
            if best_params is None:
                raise RuntimeError("La recherche n'a trouvé aucune combinaison d'hyperparamètres valide.")

            output_filename = data_processor.save_results(original_filename)

            context = data_processor.get_results_summary()
            context['quality_score'] = f"{best_score:.2%}" if best_score is not None else "N/A"
            context.update({
                'output_filename': output_filename,
                'model_type': best_params.get('model_type'),
                'best_params_found': best_params
            })
            return render_template('result.html', **context)

        except Exception as e:
            traceback.print_exc()
            flash(f"Une erreur critique est survenue : {e}")
            return redirect(request.url)

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(DOWNLOAD_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)