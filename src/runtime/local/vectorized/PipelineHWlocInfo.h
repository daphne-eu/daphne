#pragma once

#include "LoadPartitioningDefs.h"
#include <hwloc.h>
#include <runtime/local/context/DaphneContext.h>
#include <vector>

struct PipelineHWlocInfo {
    std::vector<int> physicalIds;
    std::vector<int> uniqueThreads;
    std::vector<int> responsibleThreads;
    bool hwloc_initialized = false;
    QueueTypeOption queueSetupScheme = QueueTypeOption::CENTRALIZED;

    PipelineHWlocInfo(DaphneContext *ctx) : PipelineHWlocInfo(ctx->getUserConfig().queueSetupScheme) {}
    PipelineHWlocInfo(QueueTypeOption qss) : queueSetupScheme(qss) { get_topology(); }

    void hwloc_recurse_topology(hwloc_topology_t topo, hwloc_obj_t obj, unsigned int parent_package_id) {
        if (obj->type != HWLOC_OBJ_CORE) {
            for (unsigned int i = 0; i < obj->arity; i++) {
                hwloc_recurse_topology(topo, obj->children[i], parent_package_id);
            }
        } else {
            physicalIds.push_back(parent_package_id);
            for (unsigned int i = 0; i < obj->arity; i++)
                uniqueThreads.push_back(obj->children[i]->os_index);

            switch (queueSetupScheme) {
            case QueueTypeOption::CENTRALIZED: {
                responsibleThreads.push_back(0);
            } break;
            case QueueTypeOption::PERGROUP: {
                if (responsibleThreads.size() == parent_package_id)
                    responsibleThreads.push_back(obj->children[0]->os_index);
            } break;
            case QueueTypeOption::PERCPU: {
                responsibleThreads.push_back(obj->os_index);
            } break;
            }
        }
    }

    void get_topology() {
        if (hwloc_initialized) {
            return;
        }

        hwloc_topology_t topology = nullptr;

        hwloc_topology_init(&topology);
        hwloc_topology_load(topology);

        hwloc_obj_t package = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_PACKAGE, nullptr);

        while (package != nullptr) {
            auto package_id = package->os_index;
            hwloc_recurse_topology(topology, package, package_id);
            package = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_PACKAGE, package);
        }

        hwloc_topology_destroy(topology);
        hwloc_initialized = true;
    }
};
