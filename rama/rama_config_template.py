from config_manager import config_field, config_template
from rama import constants


class RamaConfigTemplate:

    _tasks_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.INDICES,
                types=[list],
                requirements=[
                    lambda x: all([isinstance(i, list) for i in x]),
                    lambda x: all([all([isinstance(j, int) for j in i]) for i in x]),
                    lambda x: all([all([j >= 0 and j < 10 for j in i]) for i in x]),
                ],
            ),
            config_field.Field(
                name=constants.MIXING,
                types=[list],
                requirements=[lambda x: all([j >= 0 and j <= 1 for j in x])],
            ),
        ],
        # nested_templates=[_indices_template],
        level=[constants.TASKS],
    )

    _training_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.TOTAL_EPOCHS, types=[int], requirements=[lambda x: x > 0]
            ),
            config_field.Field(
                name=constants.SWITCH_EPOCH, types=[int], requirements=[lambda x: x > 0]
            ),
            config_field.Field(
                name=constants.LEARNING_RATE,
                types=[int, float],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.BATCH_SIZE, types=[int], requirements=[lambda x: x > 0]
            ),
        ],
        level=[constants.TRAINING],
    )

    _network_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.INPUT_DIMENSION,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.HIDDEN_DIMENSION,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.OUTPUT_DIMENSION,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.NONLINEARITY,
                types=[str],
                requirements=[lambda x: x in [constants.RELU, constants.SIGMOID]],
            ),
        ],
        level=[constants.NETWORK],
    )

    _plotting_template = config_template.Template(
        fields=[
            config_field.Field(name=constants.XLABEL, types=[str]),
            config_field.Field(name=constants.SMOOTHING, types=[int]),
        ],
        level=[constants.PLOTTING],
    )

    base_template = config_template.Template(
        fields=[
            config_field.Field(name=constants.SEED, types=[int]),
            config_field.Field(name=constants.GPU_ID, types=[int, type(None)]),
        ],
        nested_templates=[
            _tasks_template,
            _training_template,
            _network_template,
            _plotting_template,
        ],
    )
