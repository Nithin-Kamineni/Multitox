/* eslint-disable */

import * as React from 'react';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemText from '@mui/material/ListItemText';
import Popover from '@mui/material/Popover';
import Typography from '@mui/material/Typography';
import Link from '@mui/material/Link';
import { ParametersContext } from 'context';
import {makeStyles} from '@mui/styles';
// ==============================|| LOADER ||============================== //

const useStyles = makeStyles(theme => ({
    popover: {
      pointerEvents: 'none',
    },
    popoverContent: {
      pointerEvents: 'auto',
    },
  }));

const OutputList = () => {
    const classes = useStyles();

    const { outputList } = React.useContext(ParametersContext);

    const [anchorEl, setAnchorEl] = React.useState(null);
    const [hoveredItem, setHoveredItem] = React.useState(null);
    const [keepPopOpen, setKeepPopOpen] = React.useState(false);
    const popoverRef = React.useRef(null);

    const handleKeepPopOpen = (event, item, popstatus) => {
        event.stopPropagation();
        setAnchorEl(event.currentTarget);
            setHoveredItem({
                doi: `https://doi.org/${item[0]}`,
                title: item[1],
                abstract: item[2],
                publisher: item[3],
                authors: item[4],
                plasma: item[5],
                chemicals: item[7],
                species: item[8],
                organs: item[9],
            });
        setKeepPopOpen(popstatus);
    }

    const handlePopoverOpen = (event, item) => {
        if(!keepPopOpen){
            setAnchorEl(event.currentTarget);
            setHoveredItem({
                doi: `https://doi.org/${item[0]}`,
                title: item[1],
                abstract: item[2],
                publisher: item[3],
                authors: item[4],
                plasma: item[5],
                chemicals: item[7],
                species: item[8],
                organs: item[9],
                year: item[10]
            });
        }
    };

    const handlePopoverClose = () => {
        if(!keepPopOpen){
            setAnchorEl(null);
            setHoveredItem(null);
        }
    };

    const handleDocumentClick = (event) => {
        if (popoverRef.current && !popoverRef.current.contains(event.target)) {
            if (keepPopOpen) {
                setAnchorEl(null);
                setHoveredItem(null);
                setKeepPopOpen(false);
            }
        }
    };

    React.useEffect(() => {
        document.addEventListener('click', handleDocumentClick);

        return () => {
            document.removeEventListener('click', handleDocumentClick);
        };
    }, [keepPopOpen]);

    const open = Boolean(anchorEl);

    return (
        <div style={{ marginTop: "125px", marginLeft: "20px" }}>
            outputoutputo
        </div>
    );
}

export default OutputList;
