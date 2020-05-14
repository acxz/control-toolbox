/**********************************************************************************************************************
This file is part of the Control Toolbox (https://github.com/ethz-adrl/control-toolbox), copyright by ETH Zurich.
Licensed under the BSD-2 license (see LICENSE file in main directory)
**********************************************************************************************************************/

#pragma once

#include <ct/optcon/solver/NLOptConSettings.hpp>

#include <ct/optcon/problem/QQOCProblem.hpp>

namespace ct {
namespace optcon {


// TODO should this be in its own qqp folder?

/*!
 * Base class for solvers to solve an QQOCProblem
 * (both constrained / unconstrained, etc.)
 *
 * \todo uncouple from NLOptConSettings
 */
template <size_t STATE_DIM, size_t CONTROL_DIM, typename SCALAR = double>
class QQOCSolver
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef QQOCProblem<STATE_DIM, CONTROL_DIM, SCALAR> QQOCProblem_t;

    /*!
	 * Constructor. Initialize by handing over an QQOCProblem, or otherwise by calling setProblem()
	 * @param qqocProblem shared_ptr to the QQOCProblem to be solved.
	 */
    QQOCSolver(const std::shared_ptr<QQOCProblem_t>& qqocProblem = nullptr) : qqocProblem_(qqocProblem) {}

    virtual ~QQOCSolver() = default;

    /*!
	 * set a new problem
	 * update the shared_ptr to the QQOCProblem instance and call initialize instance deriving from this class.
	 * @param qqocProblem
	 */
    void setProblem(std::shared_ptr<QQOCProblem_t> qqocProblem)
    {
        setProblemImpl(qqocProblem);
        qqocProblem_ = qqocProblem;
    }

    virtual void configure(const NLOptConSettings& settings) = 0;

    //! setup and configure the box constraints
    // return true if configuration changed, otherwise false
    virtual bool configureInputBoxConstraints(std::shared_ptr<QQOCProblem<STATE_DIM, CONTROL_DIM>> qqocProblem)
    {
        throw std::runtime_error("input box constraints are not available for this solver.");
    }

    // return true if configuration changed, otherwise false
    virtual bool configureStateBoxConstraints(std::shared_ptr<QQOCProblem<STATE_DIM, CONTROL_DIM>> qqocProblem)
    {
        throw std::runtime_error("state box constraints are not available for this solver.");
    }

    //! setup and configure the general (in)equality constraints
    // return true if configuration changed, otherwise false
    virtual bool configureGeneralConstraints(std::shared_ptr<QQOCProblem<STATE_DIM, CONTROL_DIM>> qqocProblem)
    {
        throw std::runtime_error("general constraints are not available for this solver.");
    }

    //! a method reserved for memory allocation (e.g. required for HPIPM)
    virtual void initializeAndAllocate() = 0;

    //! solve the QQOC problem
    virtual void solve() = 0;

    virtual void solveSingleStage(int N)
    {
        throw std::runtime_error("solveSingleStage not available for this solver.");
    }

    //! extract the solution (can be overriden if additional extraction steps required in specific solver)
    virtual void computeStatesAndControls() = 0;
    //! return solution for state
    const ct::core::StateVectorArray<STATE_DIM, SCALAR>& getSolutionState() { return x_sol_; }
    //! return solution for control
    const ct::core::ControlVectorArray<CONTROL_DIM, SCALAR>& getSolutionControl() { return u_sol_; }

    //! return TVLQR feedback matrices
    virtual void computeFeedbackMatrices() = 0;
    const ct::core::FeedbackArray<STATE_DIM, CONTROL_DIM, SCALAR>& getSolutionFeedback() { return L_; }

    //! compute iLQR-style lv
    virtual void compute_lv() = 0;
    //! return iLQR-style feedforward lv
    virtual const ct::core::ControlVectorArray<CONTROL_DIM, SCALAR>& get_lv() { return lv_; }

    //! return the smallest eigenvalue
    virtual SCALAR getSmallestEigenvalue()
    {
        throw std::runtime_error("getSmallestEigenvalue not available for this solver.");
    }


protected:
    virtual void setProblemImpl(std::shared_ptr<QQOCProblem_t> qqocProblem) = 0;

    std::shared_ptr<QQOCProblem_t> qqocProblem_;

    core::StateVectorArray<STATE_DIM, SCALAR> x_sol_;            // solution in x
    core::ControlVectorArray<CONTROL_DIM, SCALAR> u_sol_;        // solution in u
    ct::core::FeedbackArray<STATE_DIM, CONTROL_DIM, SCALAR> L_;  // solution feedback
    ct::core::ControlVectorArray<CONTROL_DIM, SCALAR> lv_;       // feedforward increment (iLQR-style)
};

}  // namespace optcon
}  // namespace ct
