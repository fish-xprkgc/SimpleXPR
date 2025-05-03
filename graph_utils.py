from typing import List, Tuple, Dict, Set, DefaultDict
import igraph as ig
from collections import defaultdict, deque
import os
from pathlib import Path
import pickle


class GraphManager:
    def __init__(self):
        self.graph = ig.Graph(directed=True)

        # 核心存储结构
        self.node_id_map: Dict[str, int] = {}  # {customid: igraph顶点ID}
        self.node_info: Dict[str, Dict] = {}  # {customid: {name, description}}
        self.adjacency = defaultdict(self._inner_defaultdict_factory)
        self.reverse_adjacency = defaultdict(self._inner_defaultdict_factory)
        self.edge_list: List = []
        self.triple_index = {}

    @staticmethod
    def _inner_defaultdict_factory():
        """可序列化的嵌套字典工厂函数"""
        return defaultdict(set)

    def save_pickle(self, filepath: str) -> None:
        """
        安全存储整个GraphManager对象到pickle文件
        自动创建目录并添加.pkl后缀（如未指定）
        """
        path = Path(filepath)
        if path.suffix.lower() != '.pkl':
            path = path.with_suffix('.pkl')
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_pickle(cls, filepath: str) -> 'GraphManager':
        """
        从pickle文件加载并返回新实例
        自动处理文件路径和版本兼容
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Pickle file {path} does not exist")

        with open(path, 'rb') as f:
            instance = pickle.load(f)
            if not isinstance(instance, cls):
                raise TypeError("Loaded object is not a GraphManager instance")
            return instance

    def _update_adjacency(self, src: str, dst: str, edge_name: str) -> None:
        """更新邻接关系缓存"""
        self.adjacency[src][edge_name].add(dst)
        self.reverse_adjacency[dst][edge_name].add(src)

    def add_node(self, customid: str, name: str, description: str) -> None:
        """添加节点并维护所有相关结构"""
        if customid not in self.node_id_map:
            # 更新igraph
            vid = self.graph.add_vertex().index
            self.graph.vs[vid]['customid'] = customid
            self.graph.vs[vid]['name'] = name
            self.graph.vs[vid]['description'] = description

            # 更新缓存
            self.node_id_map[customid] = vid
            self.node_info[customid] = {
                'name': name,
                'description': description
            }

    def add_edge(self, src_id: str, dst_id: str, edge_name: str) -> None:
        """添加边并维护所有相关结构"""
        if src_id in self.node_id_map and dst_id in self.node_id_map:
            triple = (src_id, edge_name, dst_id)
            # 更新igraph
            src_vid = self.node_id_map[src_id]
            dst_vid = self.node_id_map[dst_id]
            self.graph.add_edge(src_vid, dst_vid)
            eid = self.graph.ecount() - 1
            self.graph.es[eid]['name'] = edge_name
            # 更新缓存
            self.edge_list.append(triple)
            self._update_adjacency(src_id, dst_id, edge_name)

    # 新增查询方法
    def get_node_info(self, customid: str) -> Dict[str, str]:
        """直接获取节点信息"""
        return self.node_info.get(customid, {})

    def get_outgoing_edges(self, customid: str) -> Dict[str, Set[str]]:
        """获取节点的所有出边关系"""
        return dict(self.adjacency.get(customid, {}))

    def get_incoming_edges(self, customid: str) -> Dict[str, Set[str]]:
        """获取节点的所有入边关系"""
        return dict(self.reverse_adjacency.get(customid, {}))

    # 优化后的路径查询方法
    def find_paths(self, source_id: str, max_hops: int) -> List[List[Tuple]]:
        """使用缓存结构进行路径查找"""
        if source_id not in self.node_id_map:
            return []

        paths = []
        self._dfs_path_finder(
            current_id=source_id,
            remaining_hops=max_hops,
            current_path=[],
            visited_edges=set(),
            paths=paths
        )
        return paths

    def get_shortest_path(self, source_id: str, target_id: str, rel: str):
        triple = (source_id, rel, target_id)
        weights = [1] * self.graph.ecount()
        try:
            edge_index = self.triple_index[triple]
            weights[edge_index] = float('inf')
        except:
            pass
        """基于igraph内置算法的高效实现"""
        src_vid = self.node_id_map[source_id]
        dst_vid = self.node_id_map[target_id]
        # 使用OUT模式遵循有向边方向
        paths = self.graph.get_shortest_paths(
            src_vid, dst_vid, weights=weights, mode=ig.OUT, output="epath"
        )
        if not paths or not paths[0]:
            return -1, []
        current_path = paths[0]
        current_path = [self.edge_list[edge] for edge in current_path]
        if current_path[0] == triple:
            return -1, []
        return len(current_path) - 1, current_path  # 节点数-1为边数

    def get_all_entities(self) -> List[str]:
        """获取所有节点的customid"""
        return list(self.node_id_map.keys())

    def get_all_relations(self) -> List[str]:
        """获取所有边的名称"""
        return list({triple[1] for triple in self.edge_triples})

    def get_all_triples(self) -> List[Tuple[str, str, str]]:
        """获取所有三元组（src_id, edge_name, dst_id）"""
        return list(self.edge_triples)

    def _dfs_path_finder(self, current_id: str, remaining_hops: int,
                         current_path: List[Tuple], visited_edges: Set[int],
                         paths: List[List[Tuple]]) -> None:
        """基于缓存结构的深度优先路径发现"""
        if remaining_hops == 0:
            paths.append(current_path.copy())
            return

        for edge_name, destinations in self.adjacency[current_id].items():
            for dst_id in destinations:
                edge = (current_id, edge_name, dst_id)
                edge_signature = hash(edge)  # 简单边去重

                if edge_signature in visited_edges:
                    continue

                visited_edges.add(edge_signature)
                current_path.append(edge)

                self._dfs_path_finder(
                    current_id=dst_id,
                    remaining_hops=remaining_hops - 1,
                    current_path=current_path,
                    visited_edges=visited_edges,
                    paths=paths
                )

                visited_edges.remove(edge_signature)
                current_path.pop()

    def update_node_description(self, customid: str, new_description: str) -> None:
        """修改指定节点的描述信息，其他属性保持不变"""
        if customid in self.node_id_map:
            # 更新内存中的节点信息
            self.node_info[customid]["description"] = new_description

            # 更新igraph中的节点属性
            vid = self.node_id_map[customid]
            self.graph.vs[vid]["description"] = new_description


_gm_instance = None


def get_graph_manager(path: str = "") -> GraphManager:
    global _gm_instance
    if path and Path(path).exists():
        _gm_instance = GraphManager.load_pickle(path)  # 优先加载pickle
        _gm_instance.triple_index = {triple: idx for idx, triple in enumerate(_gm_instance.edge_list)}
    else:
        if _gm_instance is None:
            _gm_instance = GraphManager()

    return _gm_instance
