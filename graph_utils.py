from typing import List, Tuple, Dict, Set, DefaultDict
import igraph as ig
from collections import defaultdict
import os
from pathlib import Path
import pickle


class GraphManager:
    def __init__(self):
        self.graph = ig.Graph(directed=True)

        # 核心存储结构
        self.node_id_map: Dict[str, int] = {}  # {customid: igraph顶点ID}
        self.node_info: Dict[str, Dict] = {}  # {customid: {name, description}}
        self.edge_triples: Set[Tuple[str, str, str]] = set()
        self.adjacency = defaultdict(self._inner_defaultdict_factory)
        self.reverse_adjacency = defaultdict(self._inner_defaultdict_factory)

    @staticmethod
    def _inner_defaultdict_factory():
        """可序列化的嵌套字典工厂函数"""
        return defaultdict(set)

    '''
    def save_gml(self, filepath: str) -> None:
        """保存图结构到GML文件"""
        self.graph.write_gml(filepath)

    def load_gml(self, filepath: str) -> None:
        """从GML文件加载并重建所有数据结构"""
        self.graph = ig.Graph.Read_GML(filepath)
        self._rebuild_data_structures()

    def _rebuild_data_structures(self) -> None:
        """统一重建所有数据存储结构"""
        # 清空所有存储
        self.node_id_map.clear()
        self.node_info.clear()
        self.edge_triples.clear()
        self.adjacency.clear()
        self.reverse_adjacency.clear()

        # 重建节点信息
        for v in self.graph.vs:
            customid = v['customid']
            self.node_id_map[customid] = v.index
            self.node_info[customid] = {
                'name': v['name'],
                'description': v['description']
            }

        # 重建边信息
        vid_to_custom = {v.index: v['customid'] for v in self.graph.vs}
        for e in self.graph.es:
            src = vid_to_custom[e.source]
            dst = vid_to_custom[e.target]
            edge_name = e['name']

            self.edge_triples.add((src, edge_name, dst))
            self._update_adjacency(src, dst, edge_name)

   '''

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
            if triple not in self.edge_triples:
                # 更新igraph
                src_vid = self.node_id_map[src_id]
                dst_vid = self.node_id_map[dst_id]
                self.graph.add_edge(src_vid, dst_vid)
                eid = self.graph.ecount() - 1
                self.graph.es[eid]['name'] = edge_name

                # 更新缓存
                self.edge_triples.add(triple)
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
    if _gm_instance is None:
        if path and Path(path).exists():
            _gm_instance = GraphManager.load_pickle(path)  # 优先加载pickle
        else:
            _gm_instance = GraphManager()
    return _gm_instance
