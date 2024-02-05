# Example tutorials for running distributed jobs in Cyfronet SLURM cluster <br>
# using torch library. <br>

To run the code, simply copy the files and create the proper directories <br>
on the cluster (including dataset and environment). <br>
Then, submit the job using the following command: <br>
```bash
sbatch meetween_example_multinode.sh
``` <br>
for the multinode example, or <br>
```bash
sbatch meetween_example_single_node.sh
``` <br>