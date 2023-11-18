import click

from rna_model import (check_sequence_length_to_reactivitiy_columns,
                       remove_incomplete_sequences, save_reactivity_reactivity_error,
                       dump_as_tfrecord,read_tfrecord, train_base_model)

@click.group
def main():
    pass

main.add_command(remove_incomplete_sequences)
main.add_command(check_sequence_length_to_reactivitiy_columns)
main.add_command(save_reactivity_reactivity_error)
main.add_command(dump_as_tfrecord)
main.add_command(read_tfrecord)
main.add_command(train_base_model)


if __name__ == "__main__":
    # ps aux --sort=-%mem | head
    main()