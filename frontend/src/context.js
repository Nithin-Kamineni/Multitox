/* eslint-disable */
import PropTypes from 'prop-types';
import { createContext, useState, useEffect } from 'react';
import axios from 'axios';

// initial state
const initialState = {
    test: 1
};

// ==============================|| CONFIG CONTEXT & PROVIDER ||============================== //

const ParametersContext = createContext(initialState);

function ParamProvider({ children }) {

    // const url = "http://localhost:8022";
    const url = "https://multitox.phhp.ufl.edu/api";
    const [Catagory, setCatagory] = useState('species');


    return (
        <ParametersContext.Provider
            value={{
                url,
                Catagory,
                setCatagory,
            }}
        >
            {children}
        </ParametersContext.Provider>
    );
}

export { ParamProvider, ParametersContext };
