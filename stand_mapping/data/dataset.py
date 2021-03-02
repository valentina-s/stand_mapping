import os
import numpy as np
import rasterio
from rasterio import windows
import torch
from torch.utils.data import Dataset


class SemanticDataset(Dataset):
    def __init__(self, root, dataframe, raw_chip_size,
                 transform=None, target_transform=None,
                 use_layers=None, boundary_class=False):
        """Initialize a SemanticDataset for semantic segmentation.

        Parameters
        ----------
        root : str
          path to root of data
        dataframe : Pandas DataFrame
          dataframe containing attributes of samples to load
        raw_chip_size : int
          height and width of area to read from each input and target layer
        transform, target_transform : callable, optional
          a function/transform that takes in the input or target and returns
          a transformed version.
        use_layers : dict, optional
          key, value pairs where each key is a type of input layer and value
          is whether or not to include this layer in the stack of input layers;
          each layer requested must have a {layer}_PATH column in the
          dataframe. If use_layers is not provided, by default only NAIP 4-band
          imagery will be loaded.
        boundary_class : bool, optional
          if True, an additional semantic class will be added to the semantic
          target which indicates whether or not a pixel is a boundary between
          land cover instances.
        """
        super(SemanticDataset, self).__init__()
        self.root = root
        self.df = dataframe
        self.raw_chip_size = raw_chip_size
        self.transform = transform
        self.target_transform = target_transform
        self.path_cols = [col for col in dataframe.columns if '_PATH' in col]
        self.layer_types = [col.split('_PATH')[0].lower() for col in
                            self.path_cols]
        self.boundary_class = boundary_class

        if use_layers is None:
            self.use_layers = {layer_type: {'use': False, 'col': path_col} for
                               layer_type, path_col in
                               zip(self.layer_types, self.path_cols)}
            self.use_layers['naip']['use'] = True
        else:
            self.use_layers = {key: {'use': value} for
                               key, value in use_layers.items()}
            for layer_type in self.use_layers:
                if (layer_type.upper() + '_PATH') not in self.path_cols:
                    raise ValueError(f'Unrecognized layer type: {layer_type}')
                else:
                    self.use_layers[layer_type]['col'] = layer_type.upper() + \
                                                         '_PATH'

    def __getitem__(self, index):
        """Fetch a sample from the dataset.

        Parameters
        ----------
        index : int
          index of the sample in the dataframe to retrieve data for

        Returns
        -------
        (input, target, nolabel) : tuple
          input image as a FloatTensor; semantic segmentation target as a
          LongTensor; nolabel as a BoolTensor indicating areas that do not have
          a land cover label assigned (either because the areas was unmapped or
          because the annotator did not assign a land cover type).
        """
        window = None
        inputs = []
        for layer_type in self.use_layers:
            if self.use_layers[layer_type]['use']:
                col = self.use_layers[layer_type]['col']
                path = os.path.join(self.root, self.df.loc[index, col])
                with rasterio.open(path) as src:
                    if window is None:
                        height, width = src.shape
                        col_off = \
                            np.random.randint(0, width - self.raw_chip_size)
                        row_off = \
                            np.random.randint(0, height - self.raw_chip_size)
                        window = windows.Window(col_off, row_off,
                                                self.raw_chip_size,
                                                self.raw_chip_size)
                    img = src.read(window=window)
                    inputs.append(img)

        input = np.vstack(inputs)
        input = torch.FloatTensor(input)

        sem_path = os.path.join(self.root, self.df.loc[index, 'SEMANTIC_PATH'])
        with rasterio.open(sem_path) as src:
            sem = src.read(window=window)

        sem = torch.LongTensor(sem)
        if self.boundary_class:
            bnd_path = os.path.join(self.root,
                                    self.df.loc[index, 'BOUNDARY_PATH'])
            with rasterio.open(bnd_path) as src:
                bnd = src.read(1, window=window)
            sem[:, bnd == 1] = 6
        # 0 means no cover type assigned, 255 means area wasn't mapped
        nolabel = torch.BoolTensor(sem == 0) + torch.BoolTensor(sem == 255)
        sem[nolabel] = 255  # set all nodata values to 255

        target = sem - 1  # change to zero for first semantic label

        if self.transform:
            input = self.transform(input)
        if self.target_transform:
            target = self.target_transform(target)
            nolabel = self.target_transform(nolabel)

        return input, target, nolabel

    def __len__(self):
        return len(self.df)


class SemanticAndWatershedDataset(SemanticDataset):
    def __init__(self, root, dataframe, raw_chip_size,
                 transform=None, target_transform=None,
                 use_layers=None, boundary_class=False,
                 clip_watershed=-100):
        """Initialize a Dataset for semantic segmentation and watershed energy
        modeling. Semantic layer includes land cover types plus an optional
        cover type for boundaries between land cover objects/instances. The
        watershed energy layer indicates the distance of a pixel from the
        nearest boundary.

        Parameters
        ----------
        root : str
          path to root of data
        dataframe : Pandas DataFrame
          dataframe containing attributes of samples to load
        raw_chip_size : int
          height and width of area to read from each input and target layer
        transform, target_transform : callable, optional
          a function/transform that takes in the input or target and returns
          a transformed version.
        use_layers : dict, optional
          key, value pairs where each key is a type of input layer and value
          is whether or not to include this layer in the stack of input layers;
          each layer requested must have a {layer}_PATH column in the
          dataframe. If use_layers is not provided, by default only NAIP 4-band
          imagery will be loaded.
        boundary_class : bool, optional
          if True, an additional semantic class will be added to the semantic
          target which indicates whether or not a pixel is a boundary between
          land cover instances.
        clip_watershed : numeric, optional
          value to clip watershed energy target to. Watershed energy indicates
          -1 times the distance to the nearest boundary. The default value of
          -100 means that all pixels further than 100 meters will be treated as
          if they were only 100 meters away.
        """
        super(SemanticDataset, self).__init__()
        self.root = root
        self.df = dataframe
        self.raw_chip_size = raw_chip_size
        self.transform = transform
        self.target_transform = target_transform
        self.path_cols = [col for col in dataframe.columns if '_PATH' in col]
        self.layer_types = [col.split('_PATH')[0].lower() for col in
                            self.path_cols]
        self.boundary_class = boundary_class
        self.clip_watershed = clip_watershed

        if use_layers is None:
            self.use_layers = {layer_type: {'use': False, 'col': path_col} for
                               layer_type, path_col in
                               zip(self.layer_types, self.path_cols)}
            self.use_layers['naip']['use'] = True
        else:
            self.use_layers = {key: {'use': value} for
                               key, value in use_layers.items()}
            for layer_type in self.use_layers:
                if (layer_type.upper() + '_PATH') not in self.path_cols:
                    raise ValueError(f'Unrecognized layer type: {layer_type}')
                else:
                    self.use_layers[layer_type]['col'] = layer_type.upper() + \
                                                         '_PATH'

    def __getitem__(self, index):
        """Fetch a sample from the dataset.

        Parameters
        ----------
        index : int
          index of the sample in the dataframe to retrieve data for

        Returns
        -------
        (input, sem, watershed, nolabel, nodata) : tuple
          input image as a FloatTensor; semantic segmentation target as a
          LongTensor; watershed energy target as a LongTensor, nolabel as a
          BoolTensor indicating areas that have been delineated but did not
          have a cover type assigned; and nodata as a BoolTensor indicating
          areas that have not been mapped at all.
        """
        window = None
        inputs = []
        for layer_type in self.use_layers:
            if self.use_layers[layer_type]['use']:
                col = self.use_layers[layer_type]['col']
                path = os.path.join(self.root, self.df.loc[index, col])
                with rasterio.open(path) as src:
                    if window is None:
                        height, width = src.shape
                        col_off = \
                            np.random.randint(0, width - self.raw_chip_size)
                        row_off = \
                            np.random.randint(0, height - self.raw_chip_size)
                        window = windows.Window(col_off, row_off,
                                                self.raw_chip_size,
                                                self.raw_chip_size)
                    img = src.read(window=window)
                    inputs.append(img)

        input = np.vstack(inputs)
        input = torch.FloatTensor(input)

        sem_path = os.path.join(self.root, self.df.loc[index, 'SEMANTIC_PATH'])
        with rasterio.open(sem_path) as src:
            sem = src.read(window=window)

        sem = torch.LongTensor(sem)
        if self.boundary_class:
            bnd_path = os.path.join(self.root,
                                    self.df.loc[index, 'BOUNDARY_PATH'])
            with rasterio.open(bnd_path) as src:
                bnd = src.read(1, window=window)
            sem[:, bnd == 1] = 6
        # 0 means no cover type assigned, 255 means area wasn't mapped
        nolabel = torch.BoolTensor(sem == 0) + torch.BoolTensor(sem == 255)
        nodata = torch.BoolTensor(sem == 255)
        sem[nolabel] = 255  # set all nodata values to 255

        sem = sem - 1  # change to zero for first semantic label

        watershed_path = os.path.join(self.root,
                                      self.df.loc[index, 'WATERSHED_PATH'])
        with rasterio.open(watershed_path) as src:
            watershed = np.expand_dims(src.read(1, window=window), 0)
            if self.clip_watershed is not None:
                watershed = watershed.clip(self.clip_watershed,0)
        watershed = torch.LongTensor(watershed)


        if self.transform:
            input = self.transform(input)
        if self.target_transform:
            sem = self.target_transform(sem)
            watershed = self.target_transform(watershed)
            nolabel = self.target_transform(nolabel)
            nodata = self.target_transform(nodata)

        return input, sem, watershed, nolabel, nodata

    def __len__(self):
        return len(self.df)
