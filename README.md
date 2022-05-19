# Bachelor's Project
*Repository containing source code and results from my Bachelor's Project.*

### General
For my Bachelor's Project I am investigating whether transfer learning allows ViTs to perform on par with CNNs when small datasets and/or only limited computational resources are available for training. I am, however, restricting this research problem to a single dataset (The Rijksmuseum Challenge Dataset -- see paper), providing art classification tasks.

**WARNING** I originally planned on only classifying materials, but later decided to extend the project to different classification tasks as well. As a result, the word 'material' is often found in the source code, as variable name or in comments, while in practice this refers to other classification tasks as well.

### Directory structure
 * **writing** contains all my writing in the form of LaTeX files.
 * **data_prep** contains scripts to generate a nice (balanced) subset from the raw *Rijsmuseum Challenge* dataset, and split it up into a training, validation, and testing set. Moreover, all needed files get packed together in a tarball.
 * **results_analysis** contains scripts to process the raw data gotten from training and testing. It also includes scripts for plotting
 * **annotations** contains some example csv-files with (jpg-file, target_y) pairs.
 * **result** contains processed results, figures, etc.
 * **rijks_torch** is a Python package. This is where all the magic happens!
     * **data_loading** contains source files to construct a PyTorch dataset from the csv- and jpg-files. It also describes a class encapsulating a whole training problem, i.e. dataloaders for training, validating, and testing.
     * **training** contains my training (+ validating) and testing functions.
     * **learning_problems** contains functions to load and alter pre-trained models used in my project.
