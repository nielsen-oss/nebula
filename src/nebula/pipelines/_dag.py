"""Generate the dag from the pipeline."""

from enum import Enum
from typing import Optional, Union

from nebula.pipelines.pipe_aux import *
from nebula.pipelines.transformer_type_util import is_transformer

__all__ = ["create_dag", "print_dag"]


def create_dag(pipeline, df_input_name: str, df_output_name: str):
    """Create the dag from a pipeline."""
    dag = _DAG(pipeline, df_input_name, df_output_name)
    return dag


class _DAG:
    def __init__(self, pipeline, df_input_name: str, df_output_name):
        """Dag constructor."""
        self.dag: list[Union[Node, dict]] = []
        self._node_counter: int = 0
        self.__temp_parents: Optional[list[Union[Node, dict]]] = None
        self.__update_dag(
            None,
            NodeType.INPUT_DF,
            kws={"name": df_input_name},
        )
        self._create_dag(pipeline)
        self.__update_dag(
            None,
            NodeType.OUTPUT_DF,
            kws={"name": df_output_name},
        )

    def __get_prev_node(self) -> Optional[Node]:
        ret = None
        for i in self.dag[::-1]:
            if isinstance(i, Node):
                ret = i
                break
        return ret

    def __update_dag(
        self,
        o,
        t: NodeType,
        *,
        kws=None,
        parents: Union[list[Node], bool, None] = None,
        skip_node: bool = False,
    ) -> Optional[Node]:
        if skip_node:
            # Do not add pipelines as nodes, but keep track of them
            self.dag.append({"type": t, "params": kws})
            return None
        node = Node(o, t=t, n=self._node_counter, kwargs=kws)

        if parents is False:
            pass
        elif parents is None:
            if self.__temp_parents is not None:
                for parent_node in self.__temp_parents:
                    parent_node.add_child(node)
            else:
                prev_node = self.__get_prev_node()
                if prev_node is not None:
                    prev_node.add_child(node)
        else:
            for parent_node in parents:
                parent_node.add_child(node)

        self.dag.append(node)
        self._node_counter += 1  # Increase the node number
        self.__temp_parents = None
        return node

    def _create_dag(self, obj):
        """Create the pipeline tree and nodes."""
        # Check whether 'obj' is a storage request
        _storage_request: Enum = parse_storage_request(obj)
        if is_transformer(obj):
            self.__update_dag(obj, NodeType.TRANSFORMER)

        # If 'obj' is an iterable, recursively parse it
        elif isinstance(obj, (list, tuple)):
            for el in obj:
                self._create_dag(el)

        # 'obj' is a storage request
        elif _storage_request.value > 0:
            # sv: Storage value
            # st: Storage type
            if _storage_request == StoreRequest.STORE_DF:
                sv, msg = get_store_key_msg(obj)
                st = NodeType.STORE

            elif _storage_request == StoreRequest.STORE_DF_DEBUG:
                sv, msg = get_store_debug_key_msg(obj)
                st = NodeType.STORE_DEBUG

            elif _storage_request == StoreRequest.ACTIVATE_DEBUG:
                sv = True
                st = NodeType.STORAGE_DEBUG_MODE
                msg = MSG_ACTIVATE_DEBUG_MODE

            elif _storage_request == StoreRequest.DEACTIVATE_DEBUG:
                sv = False
                st = NodeType.STORAGE_DEBUG_MODE
                msg = MSG_DEACTIVATE_DEBUG_MODE

            elif _storage_request == StoreRequest.REPLACE_WITH_STORED_DF:
                sv, msg = get_replace_with_stored_df_msg(obj)
                st = NodeType.REPLACE_WITH_STORED_DF

            else:  # pragma: no cover
                raise ValueError("Unknown Enum in _StoreRequest")
            self.__update_dag(sv, st, kws={"msg": msg})

        # From now on is a <TransformerPipeline>
        # As a rule, the pipeline nodes (linear and split) have no data
        # but keep the following attributes:
        # - name: Optional[str]

        # -> for linear pipeline only
        # - branch: Union[Dict[str, str], None]

        # -> for split pipeline only
        # - split_function: Optional[Callable]
        # - splits_no_merge: Union[None, str, List[str]]
        # - cast_subset_to_input_schema: bool
        # - repartition_output_to_original: bool
        # - coalesce_output_to_original: bool
        # obj is a flat pipeline
        elif obj.get_pipe_type() == NodeType.LINEAR_PIPELINE:
            kws = {"name": obj.name, "branch": obj.branch}

            node_before_branch: Optional[Node] = self.__get_prev_node()

            # This is an abstract node, representing the whole pipeline
            self.__update_dag(obj, obj.get_pipe_type(), kws=kws, skip_node=True)

            if obj.branch:
                # -------------------------- BRANCH
                if obj.branch.get("storage"):
                    is_stored_df = True
                    branch_type = NodeType.BRANCH_SECONDARY_DF
                    # It shouldn't be connected if branch["storage"] is set, but for
                    # graphical visualization I need to co connect it somewhere.
                    # Add a node indicating that the input df for the branch is stored in NS
                else:
                    is_stored_df = False
                    branch_type = NodeType.BRANCH_PRIMARY_DF

                # Create an abstract-branch-node. If the sub-pipeline
                # originates from a stored DF, do not render/connect it later.
                node_abstract_branch = self.__update_dag(obj.branch, branch_type)

                if obj.stages:  # Create the branched stages
                    for el in obj.stages:
                        self._create_dag(el)

                last_branch_node = self.__get_prev_node()

                node_after_otherwise = None
                if obj.otherwise is not None:
                    # If the branch sub-pipelines starts from the primary DF,
                    # link the 'otherwise' pipeline to the abstract-branch-node,
                    # otherwise connect it to the previous node.
                    otherwise_parent_node = (
                        node_before_branch if is_stored_df else node_abstract_branch
                    )
                    # Create the abstract node 'otherwise' to graphically
                    # distinguish the "otherwise" sub-pipeline to the branched one.
                    self.__update_dag(
                        "Otherwise",
                        NodeType.OTHERWISE,
                        parents=[otherwise_parent_node],
                    )
                    for el in obj.otherwise.stages:
                        self._create_dag(el)

                    node_after_otherwise = self.__get_prev_node()

                if obj.branch["end"] == "dead-end":
                    if node_after_otherwise is not None:  # pragma: no cover
                        # This 'otherwise' / 'dead-end' combination is
                        # disallowed, but I configure it anyway
                        _dead_end_parent = node_after_otherwise
                    else:
                        if is_stored_df:
                            _dead_end_parent = node_before_branch
                        else:
                            _dead_end_parent = node_abstract_branch

                    # Different representations, both valid
                    self.__temp_parents = [_dead_end_parent]
                    # self.__temp_parents = parents
                else:
                    parents = [last_branch_node]
                    if node_after_otherwise is not None:
                        parents.append(node_after_otherwise)
                    else:
                        parents.append(node_abstract_branch)
                    self.__update_dag(
                        obj.branch,
                        NodeType.BRANCH_MERGE,
                        parents=parents,
                    )

            # -------------------------- APPLY TO ROWS
            elif obj.apply_to_rows:
                a2r_node = self.__update_dag(obj.apply_to_rows, NodeType.APPLY_TO_ROWS)

                node_after_a2r = None
                if obj.stages:
                    for el in obj.stages:
                        self._create_dag(el)
                    node_after_a2r = self.__get_prev_node()

                if obj.otherwise is not None:
                    # Create the abstract node 'otherwise' to graphically
                    # distinguish the "otherwise" sub-pipeline to the branched one.
                    self.__update_dag(
                        "Otherwise",
                        NodeType.OTHERWISE,
                        parents=[a2r_node],
                    )
                    for el in obj.otherwise.stages:
                        self._create_dag(el)

                    a2r_node = self.__get_prev_node()

                if obj.apply_to_rows.get("dead-end"):
                    self.__temp_parents = [a2r_node]
                else:
                    parents = [a2r_node]
                    if node_after_a2r is not None:
                        parents.append(node_after_a2r)
                    self.__update_dag(
                        obj,
                        NodeType.MERGE_APPLY_TO_ROWS,
                        parents=parents,
                    )

            else:
                for el in obj.stages:
                    self._create_dag(el)

        elif obj.get_pipe_type() == NodeType.SPLIT_PIPELINE:
            kws = {
                "name": obj.name,
                "split_function": obj.split_function,
                "splits_no_merge": obj.splits_no_merge,
                "cast_subset_to_input_schema": obj.cast_subset_to_input_schema,
                "repartition_output_to_original": obj.repartition_output_to_original,
                "coalesce_output_to_original": obj.coalesce_output_to_original,
            }
            self.__update_dag(obj, obj.get_pipe_type(), kws=kws, skip_node=True)
            split_func_node = self.__update_dag(
                obj.split_function, NodeType.SPLIT_FUNCTION
            )

            parents: list[Node] = []

            for split_name, el in obj.splits.items():
                # The parent node for each split is the split function node

                self.__update_dag(
                    split_name,
                    NodeType.SPLIT_NAME,
                    parents=[split_func_node],
                )
                self._create_dag(el)
                # parents.append(self._dag[-1])
                parents.append(self.__get_prev_node())

            _merge_kws_msg = {
                "Allow missing columns": obj.allow_missing_cols,
                "Cast each split to input schema": obj.cast_subset_to_input_schema,
            }
            if obj.repartition_output_to_original:
                _merge_kws_msg["Repartition to original"] = True
            if obj.coalesce_output_to_original:
                _merge_kws_msg["Coalesce to original"] = True
            self.__update_dag(
                _merge_kws_msg,
                NodeType.SPLIT_MERGE,
                parents=parents,
            )
