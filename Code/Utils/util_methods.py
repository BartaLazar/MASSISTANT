import json
import os
import re

import pandas as pd
from IPython.core.display_functions import display
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw, rdFingerprintGenerator
from selfies import EncoderError
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import selfies as sf



class NNUtils:

    current_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(f'{current_dir}/../../.env')

    @staticmethod
    def read_big_csv(file, chunk_size=1000, index_col=None):
        merged_chunks = []

        for chunk in tqdm(pd.read_csv(file, iterator=True, chunksize=chunk_size), desc=f'Reading CSV file {file.split("/")[-1]}'):
            merged_chunks.append(chunk)

        merged_df = pd.concat(merged_chunks, ignore_index=True)

        if index_col:
            return merged_df.set_index(index_col)

        return merged_df


    @staticmethod
    def visualize_spectra(spectra_data, unknown_spectra=False, title='Mass Spectra for multiple SMILES'):
        ''''
        Display the spectra
        :param spectra_data: A pandas DataFrame containing the spectra data. It should contain the following columns: SMILES, Mw, mz1, mz2, ..., mzN
        :param unknown_spectra: Boolean to indicate if the SMILES is unknown. SMILES and Mw columns are not required if True.
        :param title: The title of the plot
        '''
        # create figure and axis
        fig, ax = plt.subplots()

        molecular_weights = []

        for row in spectra_data.itertuples(index=False):

            try:
                smiles = row.SMILES
                print(f'{smiles} :')
            except AttributeError:
                smiles = None

            spectra = [getattr(row, f'mz{x}') for x in range(1, int(os.getenv('MAX_MASS')) + 1)]

            try:
                molecular_weight = getattr(row, 'Mw')
            except AttributeError:
                molecular_weight = None

            molecular_weights.append(molecular_weight)

            # plot bar chart with different colors for each row
            if unknown_spectra:
                bars = ax.bar(range(1, len(spectra) + 1), spectra, label='Unknown spectrum')
            else:
                bars = ax.bar(range(1, len(spectra) + 1), spectra, label=f'{row.SMILES} Mw: {molecular_weight}')
                # add vertical line for molecular weight
                color = bars[0].get_facecolor()
                ax.axvline(x=molecular_weight, color=color, linestyle='--')

        ax.set_xlabel('Mass/Charge (m/z)')
        ax.set_ylabel('Intensity')
        ax.set_title(title)
        if not unknown_spectra:
            x_max = max(
                max(i for row in spectra_data.itertuples(index=False) for i, val in enumerate([getattr(row, f'mz{x}') for x in range(1, int(os.getenv('MAX_MASS')) + 1)], start=1) if val > 0),
                max(molecular_weights)
            )
        else:
            x_max = max(i for row in spectra_data.itertuples(index=False) for i, val in enumerate([getattr(row, f'mz{x}') for x in range(1, int(os.getenv('MAX_MASS')) + 1)], start=1) if val > 0)
        ax.set_xlim(1, x_max)
        ax.legend()
        plt.show()


    @staticmethod
    def draw_molecule(smiles):
        """
        Draw a molecule from a SMILES string
        :param smiles: Smiles string of the molecule
        :return:
        """
        # Convert the SMILES string to an RDKit molecule object
        molecule = Chem.MolFromSmiles(smiles)

        # Draw the molecule and display it
        image = Draw.MolToImage(molecule)

        # Display the image
        display(image)

    @staticmethod
    def divide_big_train_and_test_data(X, y, test_size=float(os.getenv('TEST_SIZE')), random_state=42, chunk_size=10000, np_array_input=False, verbose=True, x_as_float32=True, y_as_float32=True, desc='Splitting data into chunks'):
        """
        Divide the data into training and testing sets. Can be used for any other split task.
        :param X: The features
        :param y: The target labels
        :param test_size: The proportion of the data to include in the test set
        :param random_state: The seed used by the random number generator
        :param chunk_size: The size of the chunks to split the data into
        :param np_array_input: Whether the input is a numpy array or not
        :param verbose: Whether to print the progress bar or not
        :param x_as_float32: Whether to cast the features to float32 or not
        :param y_as_float32: Whether to cast the target labels to float32 or not
        :return: The training and testing sets (X_train, X_test, y_train, y_test)
        """

        X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

        # Read in chunks
        for i in tqdm(range(0, len(X), chunk_size), desc=desc, disable=not verbose):

            if not np_array_input:
                # Split features and target label
                if i + chunk_size > len(X):
                    X_chunk = X.iloc[i:].astype('float32') if x_as_float32 else X.iloc[i:]
                    y_chunk = y.iloc[i:].astype('float32') if y_as_float32 else y.iloc[i:]
                else:
                    y_chunk = y.iloc[i:i + chunk_size].astype('float32') if y_as_float32 else y.iloc[i:i + chunk_size]
                    X_chunk = X.iloc[i:i + chunk_size].astype('float32') if x_as_float32 else X.iloc[i:i + chunk_size]
            else:
                if i + chunk_size > len(X):
                    X_chunk = X[i:]
                    y_chunk = y[i:]
                else:
                    y_chunk = y[i:i + chunk_size]
                    X_chunk = X[i:i + chunk_size]

            # Split the chunk into train and test sets
            X_train_chunk, X_test_chunk, y_train_chunk, y_test_chunk = train_test_split(
                X_chunk, y_chunk, test_size=test_size, random_state=random_state
            )

            # Append the results
            X_train_list.append(X_train_chunk)
            X_test_list.append(X_test_chunk)
            y_train_list.append(y_train_chunk)
            y_test_list.append(y_test_chunk)

        # Concatenate all chunks to form full train and test sets
        X_train = pd.concat(X_train_list)
        X_test = pd.concat(X_test_list)
        y_train = pd.concat(y_train_list)
        y_test = pd.concat(y_test_list)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def inner_join_big_dataframes(left_df, right_df, key, chunk_size=10000):
        """
        Perform an inner join on two large dataframes.
        :param left_df: The left dataframe
        :param right_df: The right dataframe
        :param key: The column to join on
        :param chunk_size: The size of the chunks to split the data into
        :return: The joined dataframe
        """
        df_joined_list = []
        for i in tqdm(range(0, len(left_df), chunk_size)):
            if i + chunk_size > len(left_df):
                df_joined = pd.merge(left_df.iloc[i:, :], right_df, on=key, how='inner')
            else:
                df_joined = pd.merge(left_df.iloc[i:i + chunk_size, :], right_df, on=key, how='inner')
            df_joined_list.append(df_joined)

        df_joined = pd.concat(df_joined_list, ignore_index=True)
        return df_joined

    @staticmethod
    def group_by_embedding(df, embedding_size, min_collisions=1, verbose=False):
        '''
        Group embeddings by their values and count the number of collisions
        :param df: DataFrame containing the embeddings
        :param embedding_size: Number of bits in the fingerprint
        :param min_collisions: Minimum number of collisions to consider
        :param verbose: Boolean to print extra insights
        :return: DataFrame containing the embeddings with different fingerprints. Columns: bit_0, bit_1, ..., bit_n, NB_collisions, SMILES_list
        '''

        subset_columns = [f'bit_{i}' for i in range(embedding_size)]

        # find values that have a collision (appear at least twice in the dataframe)
        collision_values = df[subset_columns].duplicated(keep=False)

        # separate collided and non-collided rows
        collisions = df[collision_values].copy()

        if min_collisions == 1:
            non_collisions = df[~collision_values].copy()

        # group by subset columns and collect SMILES values for collisions
        collisions['SMILES_list'] = collisions.groupby(subset_columns)['SMILES'].transform(lambda x: ', '.join(x))
        collisions['NB_collisions'] = collisions['SMILES_list'].apply(lambda x: len(x.split(', ')))
        collisions = collisions.drop_duplicates(subset=subset_columns)

        # add non-collided rows to the final dataframe
        if min_collisions == 1:
            non_collisions['SMILES_list'] = non_collisions['SMILES']
            non_collisions['NB_collisions'] = 1

        # concatenate collisions and non-collisions
        if min_collisions == 1:
            df_final = pd.concat([collisions, non_collisions], ignore_index=True)
        else:
            df_final = collisions[collisions['NB_collisions'] >= min_collisions]

        # print the different values for these subsets of columns that have collisions
        if verbose:
            print(f'Number of embeddings with different fingerprints: {len(collisions)} out of {len(df)}')
        df_collision = df_final[subset_columns + ['NB_collisions', 'SMILES_list']]

        return df_collision

    @staticmethod
    def beautify_json(json_data, sort_keys=True, indent=4):
        """
        Beautify a JSON string
        :param json_data: The JSON string or dict to beautify
        :param sort_keys: Whether to sort the keys or not
        :param indent: The number of spaces to indent
        :return: The beautified JSON string
        """
        return json.dumps(json_data, indent=indent, sort_keys=sort_keys)


    @staticmethod
    def draw_morgan_bits(embedding, all_morgan_bits, original_predicted_emb=None, bits_per_row=4):
        '''
        Draws the Morgan bits for a given embedding.
        :param embedding: Embedding to visualize. Contains bits bit_0, bit_1, ... bit_n and (Pandas Series (row of a dataframe))
        :param all_morgan_bits: Dictionary with the visual Morgan bits. The keys are bit_n and the values are the morgan bit image (Dict)
        :param original_predicted_emb: The original predicted embeddings, without threshold. Used to show the confidence of the prediction of the bits. (Pandas Series (row of a dataframe))
        :param bits_per_row: Number of bits to show per row (int)
        :return: Figure object
        '''

        embeddings = embedding[[f'bit_{i}' for i in range(int(os.getenv('EMBEDDING_SIZE')))]]

        one_bits = embeddings[embeddings == 1].index.tolist()
        morgan_bits = [all_morgan_bits[i] for i in one_bits]

        w = 20
        h = w
        columns = bits_per_row
        rows = (len(morgan_bits) // columns) + 1

        fig = plt.figure(figsize=(15, rows * 4.5))

        # setting the spacing between subplots
        fig.subplots_adjust(hspace=0, wspace=0.2)  # add vertical and horizontal spacing

        # iterate through the grid positions and only add subplots for available images
        for i in range(1, columns * rows + 1):
            ax = fig.add_subplot(rows, columns, i)

            # check if there's an image to plot
            if i - 1 < len(morgan_bits):
                img = morgan_bits[i - 1]
                # ax.size = (w, h)
                ax.imshow(img)
                if original_predicted_emb is not None:
                    # adding a confidence below the image
                    subtitle = f'Confidence: {original_predicted_emb[one_bits[i - 1]] * 100:.2f}%'
                    ax.text(0.5, -0.1, subtitle, ha='center', va='top', transform=ax.transAxes, fontsize=10)
                ax.set_title(one_bits[i - 1])  # set title for available images
            ax.axis('off')  # turning off the axis for a cleaner look

        # plt.tight_layout()
        plt.show()

        return fig

    @staticmethod
    def get_morgan_bits(df, priority_index=None, show_morgan_bits=0, bits_per_row=4):
        '''
        Get all the morgan bits from a DataFrame containing the SMILES
        :param df: DataFrame with the SMILES (Pandas DataFrame)
        :param priority_index: Index of the molecule to create the morgan bits from first. If None, it will show the morgan bits for the first molecule (int)
        :param show_morgan_bits: 0 not to show the morgan bits, 1 to show the final morgan bits, 2 to show the morgan bits for the priority index molecule (int)
        :param bits_per_row: Number of bits to show per row if the morgan bits are displayed (int)
        :return: dictionary with the graphical morgan bits. The key is the bit number (bit_n) and the value is the image
        '''

        print('Extracting Morgan bits')

        morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=int(os.getenv('RADIUS')), fpSize=int(os.getenv('EMBEDDING_SIZE')))
        all_morgan_bits = {}
        priority_done = True
        if priority_index is not None:
            priority_done = False
        for i in range(df.shape[0]):

            if not priority_done:
                k = i
                i = priority_index

            ao = rdFingerprintGenerator.AdditionalOutput()
            ao.AllocateAtomCounts()
            ao.AllocateAtomToBits()
            ao.AllocateBitInfoMap()

            mol = Chem.MolFromSmiles(df.iloc[i]['SMILES'])
            fp = morgan_generator.GetFingerprint(mol, additionalOutput=ao)
            bi = ao.GetBitInfoMap()


            for key in bi:
                if f'bit_{key}' not in all_morgan_bits:
                    len_dict = len(all_morgan_bits)
                    all_morgan_bits[f'bit_{key}'] = Draw.DrawMorganBit(mol, key, bi)

                    print(f'{len(all_morgan_bits)} / {os.getenv("EMBEDDING_SIZE")} bits extracted')

                    if len_dict == os.getenv('EMBEDDING_SIZE'):
                        print('All Morgan bits succesfully extracted')
                        return all_morgan_bits

            if not priority_done:
                priority_done = True
                i = k

        print(f'{len(all_morgan_bits)} / {os.getenv("EMBEDDING_SIZE")} Morgan bits extracted')

        if show_morgan_bits > 0:

            emb_cols = [f'bit_{i}' for i in range(int(os.getenv('EMBEDDING_SIZE')))]

            if show_morgan_bits == 2:
                NNUtils.draw_morgan_bits(df.iloc[priority_index][emb_cols], all_morgan_bits, bits_per_row=bits_per_row)
            else:
                for i in range(df.shape[0]):
                    print(f'Row {i}')
                    NNUtils.draw_morgan_bits(df.iloc[i][emb_cols], all_morgan_bits, bits_per_row=bits_per_row)

        return all_morgan_bits

    @staticmethod
    def get_morgan_bits_from_smiles(smiles):
        morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=int(os.getenv('RADIUS')),
                                                                     fpSize=int(os.getenv('EMBEDDING_SIZE')))
        morgan_bits = {}
        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateAtomCounts()
        ao.AllocateAtomToBits()
        ao.AllocateBitInfoMap()

        mol = Chem.MolFromSmiles(smiles)
        fp = morgan_generator.GetFingerprint(mol, additionalOutput=ao)
        bi = ao.GetBitInfoMap()

        for key in bi:
            morgan_bits[f'bit_{key}'] = Draw.DrawMorganBit(mol, key, bi)

        return morgan_bits


    @staticmethod
    def find_project_root(current_file, marker=".rootfolder"):
        """
        Find the project root directory containing a specific marker file.
        :param current_file: Path to the current file. Use os.getcwd() if unsure. (str)
        :param marker: File or directory name in the root folder that indicates the project root. Default is ".rootfolder" (str)
        :return: The path to the project root directory. (str)
        """
        # start with the directory containing the current file
        current_dir = os.path.abspath(os.path.dirname(current_file))

        while current_dir != os.path.dirname(current_dir):  # stop when at filesystem root
            if marker in os.listdir(current_dir):
                return current_dir
            current_dir = os.path.dirname(current_dir)  # move one directory up

        raise FileNotFoundError(f"Project root with marker '{marker}' not found.")

    @staticmethod
    def add_selfies_to_df(df_input, smiles_col='SMILES', selfies_col='SELFIES', invalid="Invalid"):
        '''
        Add a new column to the dataframe with the SELFIES representation of the SMILES strings.
        :param df_input: input dataframe
        :param smiles_col: column name with SMILES strings
        :param selfies_col: column name for the new SELFIES column
        :param invalid: value for invalid SMILES strings
        :return: new dataframe with the SELFIES column added
        '''
        df = df_input.copy()
        selfies = []
        inv = 0
        for smiles in tqdm(df[smiles_col], desc='Adding SELFIES to the dataframe'):
            # selfies.append(sf.encoder(smiles))
            try:
                selfies.append(sf.encoder(smiles))
            except EncoderError as e:
                # handle the invalid SMILES case
                selfies.append(invalid)
                inv += 1

        df[selfies_col] = selfies
        print(f'Number of invalid SMILES: {inv}')

        return df

    @staticmethod
    def selfies_similarity(selfies1, selfies2):
        '''
        Calculate the similarity between two SELFIES strings. The similarity is calculated as the number of matching parts / length of the longest SELFIES string.
        Custom made function to compare two SELFIES strings.
        :param selfies1: str, the reference SELFIES string
        :param selfies2: str, the predicted SELFIES string
        :return: float, the similarity between the two SELFIES strings
        '''
        # transform the SELFIES strings into lists of parts
        sl1 = re.findall(r'\[(.*?)\]', selfies1)
        sl2 = re.findall(r'\[(.*?)\]', selfies2)

        similarity = 0
        for i in range(len(sl1)):
            if i < len(sl2) and sl1[i] == sl2[i]:
                similarity += 1

        if len(sl1) >= len(sl2):
            return similarity / len(sl1)
        else:
            return similarity / len(sl2)

    @staticmethod
    def dataset_loader(datatype='pkl', **kwargs):
        '''
        Loads the datasets
        :param datatype: Type of dataset to load ('csv' or 'pkl')
        :param kwargs: Keyword arguments (argument name = path)
        :return: Dictionary of datasets with the provided argument names as keys
        '''
        return_dict = {}
        if datatype == 'pkl':
            for key, value in kwargs.items():
                return_dict[key] = pd.read_pickle(value)
        elif datatype == 'csv':
            for key, value in kwargs.items():
                return_dict[key] = NNUtils.read_big_csv(value)
        return return_dict