// assets
import { IconDashboard, IconGraph } from '@tabler/icons-react';

// constant
const icons = { IconDashboard, IconGraph };

// ==============================|| DASHBOARD MENU ITEMS ||============================== //

const dashboard = {
  id: 'dashboard',
  title: 'Dashboard',
  type: 'group',
  children: [
    {
      id: 'default',
      title: 'Dashboard',
      type: 'item',
      url: '/dashboard/default',
      icon: icons.IconDashboard,
      breadcrumbs: false
    },
    // {
    //   id: 'freq',
    //   title: 'Statestics',
    //   type: 'item',
    //   url: '/dashboard/stats/',
    //   icon: icons.IconGraph,
    //   breadcrumbs: false
    // }
  ]
};

export default dashboard;
