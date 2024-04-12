"""The global constants used in the project."""

# Contact Tracing Capacity (CTC) is a constant from the dpfn and abm library.
# The simulator will simulate agents in various contact patterns such as
# households, schools, workplaces, and communities. To fundamentally switch
# between graph-based models and fixed-sized arrays, this constant is used.
# CTC is the maximum number of contacts an agent can have 'in a time window,'
# where the time window is usually fourteen days. Any contacts beyond this are
# ignored based on recency. Emperically, I have found that a CTC of 900 covers
# about 99% of the agents in the population. This is a good trade-off between
# memory and accuracy and works on both the COVASIM and Oxford OpenABM models.
CTC = 900
